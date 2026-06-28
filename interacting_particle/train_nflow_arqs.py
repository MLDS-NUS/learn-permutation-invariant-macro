import os
import logging
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm

import matplotlib.pyplot as plt


# Your existing model components
from models import (
    AnchorGaussianMixtureND,
    SetEncoderND,
)
from utils.utils import set_seed


# nflows: autoregressive RQS
from nflows import transforms as nf_transforms
from nflows import distributions as nf_distributions
from nflows import flows as nf_flows


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _should_record(time_step: int) -> bool:
    i = time_step + 1
    if i <= 100:
        return True
    elif i <= 150:
        return (i % 2) == 0
    elif i <= 200:
        return (i % 5) == 0
    else:
        return (i % 10) == 0
    
        
def make_dataloaders(
    data_path: str,
    bs: int,
    n_traj: int,
):
    """
    Load training & validation anchor sets and wrap them in TensorDataset/DataLoader.

    Each file is assumed to contain a numpy array of shape [E, T, M, D], where:
      - E: number of experiments
      - T: number of timesteps
      - M: number of anchors per experiment
      - D: data dimension

    We treat each (experiment, timestep) pair as a separate "experiment":
      - Select first 100 timesteps (or all if T < 100)
      - Reshape to (E*T_sel, M, D)
    """
    
    
    input_data = np.load(data_path, allow_pickle=True)
    assert input_data.shape[0] >= n_traj, f"data has {input_data.shape[0]} trajectories, but n_traj={n_traj}"
    input_data = input_data[:n_traj]  # (E, T, M, D)
    print(f"input_data shape: {input_data.shape}")

    # To reduce training time, select timesteps according to _should_record
    selected_timesteps = [t for t in range(input_data.shape[1]) if _should_record(t)]
    input_data = input_data[:, selected_timesteps, :, :]  # (E, T_sel, M, D)
    print(f"Selected {len(selected_timesteps)} timesteps: {selected_timesteps}")
    print(f"input_data shape after timestep selection: {input_data.shape}")
        
    
    # normalize data to zero mean, 1 std in each dimension
    mean = input_data.mean(axis=(0, 1, 2), keepdims=True)
    std = input_data.std(axis=(0, 1, 2), keepdims=True)
    norm_input_data = (input_data - mean) / std
    
    normalization_info = {
        "mean": mean,
        "std": std,
    }
    print(f"Data normalization info: {normalization_info}")
    

    

    train_num = int(0.8 * norm_input_data.shape[0])
    train_pos = norm_input_data[:train_num]   # (E_train, T, M, D)
    val_pos = norm_input_data[train_num:]     # (E_val, T, M, D)
    
    # print(f"safe_ranges: {safe_ranges}")
    print(
        f"train x,y max: {train_pos.max(axis=(0, 1, 2))}, "
        f"min: {train_pos.min(axis=(0, 1, 2))}"
    )
    print(
        f"val   x,y max: {val_pos.max(axis=(0, 1, 2))}, "
        f"min: {val_pos.min(axis=(0, 1, 2))}"
    )

    

    # Reshape: (E,T_sel,M,D) -> (E*T_sel, M, D)
    train_anchors = torch.from_numpy(
        train_pos.reshape(-1, train_pos.shape[2], train_pos.shape[3])
    ).float()
        
    val_anchors = torch.from_numpy(
        val_pos.reshape(-1, val_pos.shape[2], val_pos.shape[3])
    ).float()    

    assert train_anchors.dim() == 3
    assert val_anchors.dim() == 3

    # Wrap in datasets/loaders; each item is a single experiment's anchor set: (M,D)
    train_ds = TensorDataset(train_anchors)
    val_ds = TensorDataset(val_anchors)
    print(f"train_anchors shape: {train_anchors.shape}, val_anchors shape: {val_anchors.shape}")

    train_loader = DataLoader(train_ds, batch_size=bs, pin_memory=True, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, pin_memory=True, shuffle=True, drop_last=False)

    dim = train_pos.shape[3]
    n_anchors = train_pos.shape[2]

    return train_loader, val_loader, dim, n_anchors, normalization_info


# ---------------------------------------------------------------------------
# nflows-based conditional AR RQS model
# ---------------------------------------------------------------------------
# @torch.compile
class NFlowsConditionalARQS(torch.nn.Module):
    """
    Wrapper around nflows MaskedPiecewiseRationalQuadraticAutoregressiveTransform.

    - Context z_ctx is produced by your SetEncoderND from anchors.
    - Flow base distribution is StandardNormal.
    - log_prob(x, anchors) matches the interface of your ConditionalFlowND.

    x:       (E,B,D)
    anchors: (E,M,D)
    returns: log p(x | anchors): (E,B)
    """

    def __init__(
        self,
        dim: int,
        z_dim: int,
        hidden_dim: int,
        n_layers: int,
        K: int,
        B: float,
        set_encoder: SetEncoderND,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.K = K
        self.B = B

        # Register your DeepSet encoder as a submodule so its parameters are optimized too.
        self.set_encoder = set_encoder

        # Build a stack of autoregressive RQS transforms with random permutations between them.
        transforms = []
        for _ in range(n_layers):
            transforms.append(
                nf_transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=dim,
                    hidden_features=hidden_dim,
                    context_features=z_dim,
                    num_bins=K,
                    tails="linear",      # linear tails outside [-B,B]
                    tail_bound=B,
                )
            )
            # Random permutation for extra mixing
            transforms.append(
                nf_transforms.RandomPermutation(features=dim)
            )

        transform = nf_transforms.CompositeTransform(transforms)
        base_dist = nf_distributions.StandardNormal(shape=[dim])

        self.flow = nf_flows.Flow(transform=transform, distribution=base_dist)

    def log_prob(self, x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """
        x:       (E,B,D)
        anchors: (E,M,D)
        returns: (E,B)
        """
        
        E, B, D = x.shape
        assert D == self.dim

        # DeepSet encoder to get context per experiment: (E,z_dim)
        z_ctx = self.set_encoder(anchors)  # (E,z_dim)

        # Flatten experiments + batch for the flow
        N = E * B
        x_flat = x.view(N, D)

        # Repeat context across batch dimension
        z_expanded = z_ctx.unsqueeze(1).expand(E, B, self.z_dim)  # (E,B,z_dim)
        context_flat = z_expanded.reshape(N, self.z_dim)          # (N,z_dim)

        # nflows Flow.log_prob supports a context argument. 
        log_p_flat = self.flow.log_prob(x_flat, context=context_flat)  # (N,)

        return log_p_flat.view(E, B)
    

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        anchors: torch.Tensor,
        *,
        return_log_prob: bool = False,
    ):
        """
        Sample from p(x | anchors) after training.

        Args:
            num_samples: Number of samples per anchor set (per experiment).
            anchors: Tensor of shape (E, M, D) or (M, D) for a single anchor set.
            return_log_prob: If True, also returns log p(samples | anchors).

        Returns:
            samples: (E, num_samples, D)
            (optional) log_prob: (E, num_samples)
        """
        # Allow passing a single anchor set of shape (M, D)
        if anchors.dim() == 2:
            anchors = anchors.unsqueeze(0)
        if anchors.dim() != 3:
            raise ValueError(
                f"anchors must have shape (E,M,D) or (M,D); got {tuple(anchors.shape)}"
            )
        

        # Context per experiment (E, z_dim)
        z_ctx = self.set_encoder(anchors)

        # nflows Flow.sample supports conditioning via `context=...`
        # For batched context, it returns (E, num_samples, D)
        samples = self.flow.sample(num_samples, context=z_ctx)

        if not return_log_prob:
            return samples

        log_p = self.log_prob(samples, anchors)


        return samples, log_p



# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_subset_loader(
    loader: DataLoader,
    fraction: float,
    rng: np.random.Generator,
) -> DataLoader:
    if fraction >= 1.0:
        return loader
    num_batches = len(loader)
    keep_batches = max(1, int(np.ceil(num_batches * fraction)))
    subset_size = min(len(loader.dataset), keep_batches * loader.batch_size)
    indices = rng.choice(len(loader.dataset), size=subset_size, replace=False).tolist()
    sampler = SubsetRandomSampler(indices)
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=loader.drop_last,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        collate_fn=loader.collate_fn,
    )


def train_conditional_flow_from_files(
    data_path: str,
    n_traj: int,
    model_type: str = "ARQS",   # "RQS" (your coupling flow) or "ARQS" (nflows)
    epsilon: float = 0.05,
    dim: int | None = None,
    bs: int = 32,
    samples_per_experiment: int = 512,
    val_samples_per_experiment: int = 512,
    n_layers: int = 8,
    rqs_K: int = 16,
    rqs_B: float = 10.0,
    z_dim: int = 4,
    hidden_dim: int = 256,
    deepset_pool: str = "mean",
    lr: float = 1e-3,
    num_epochs: int = 20,
    seed: int = 42,
    device: str | torch.device | None = None,
    save_path: str = "conditional_flow_nflows.pth",
    args_dict: dict | None = None,
):
    """
    Train a conditional flow on pre-generated anchor experiments.

    - model_type == "RQS": uses your existing ConditionalFlowND (coupling RQS).
    - model_type == "ARQS": uses nflows MaskedPiecewiseRationalQuadraticAutoregressiveTransform.

    Each batch from the DataLoader has shape (bs, M, D), where:
      - bs: number of experiments in the batch
      - M:  number of anchors
      - D:  data dimension
    """

    set_seed(seed)

    # Device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Data
    train_loader, val_loader, inferred_dim, n_anchors, normalization_info = make_dataloaders(
        data_path=data_path,
        bs=bs,
        n_traj=n_traj,
    )
    if dim is None:
        dim = inferred_dim
    else:
        assert dim == inferred_dim, f"dim={dim} but data has dim={inferred_dim}"

    # Logger
    log_dir = os.path.dirname(save_path) if save_path else "."
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_Z{z_dim}_{model_type}.log")
    logger = logging.getLogger(f"TrainFlow:{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # avoid duplicate handlers if called multiple times
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    print(f"Logging to {log_path}")
    logger.info("Starting training")
    logger.info(f"seed: {seed}, z_dim: {z_dim}")

    # Shared DeepSet encoder
    set_encoder = SetEncoderND(
        in_dim=dim,
        hidden_dim=hidden_dim,
        z_dim=z_dim,
        pool=deepset_pool,
    ).to(device)

    # Build model    
    # nflows autoregressive RQS, context = DeepSet(anchors)
    model = NFlowsConditionalARQS(
        dim=dim,
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        K=rqs_K,
        B=rqs_B,
        set_encoder=set_encoder,
    ).to(device)
    

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    train_loss_history = []
    val_loss_history = []
    eval_epochs = []
    train_batch_fraction = 0.25
    val_batch_fraction = 0.5
    rng = np.random.default_rng()
    # ----------------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------------
    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        train_losses = []

        train_loader_epoch = _make_subset_loader(train_loader, train_batch_fraction, rng)
        for (anchors_batch, ) in train_loader_epoch:
            anchors_batch = anchors_batch.to(device, non_blocking=True)  # (E_batch, M, D)
            E_batch, M, D = anchors_batch.shape

            # Target GMM for this batch (adaptive std via KNN)
            target_batch = AnchorGaussianMixtureND(
                anchors=anchors_batch,   # (E_batch, M, D)
                epsilon=epsilon,
            )

            # Sample from target: (E_batch, B, D)
            x = target_batch.sample(samples_per_experiment)
            # Compute log p_model(x | anchors)
            log_p_model = model.log_prob(x, anchors_batch)  # (E_batch,B)

            loss = -log_p_model.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses))

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        model.eval()
        val_losses = []
        val_loader_epoch = _make_subset_loader(val_loader, val_batch_fraction, rng)
        with torch.no_grad():
            for (anchors_batch_val,) in val_loader_epoch:
                anchors_batch_val = anchors_batch_val.to(device, non_blocking=True)

                target_val = AnchorGaussianMixtureND(
                    anchors=anchors_batch_val,   # (E_batch, M, D)
                    epsilon=epsilon,
                )
                x_val = target_val.sample(val_samples_per_experiment)
                log_p_val = model.log_prob(x_val, anchors_batch_val)

                val_loss = -log_p_val.mean()
                val_losses.append(val_loss.item())

        mean_val_loss = float(np.mean(val_losses))

        msg = (
            f"Epoch {epoch:4d} | "
            f"Train NLL: {mean_train_loss:8.4f} | "
            f"Val NLL: {mean_val_loss:8.4f}"
        )
        train_loss_history.append(mean_train_loss)
        val_loss_history.append(mean_val_loss)
        eval_epochs.append(epoch)
        # print(msg)
        logger.info(msg)

        # Save best model
        if mean_val_loss < best_val_loss and save_path is not None:
            best_val_loss = mean_val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "dim": dim,
                "epsilon": epsilon,
                "n_layers": n_layers,
                "z_dim": z_dim,
                "hidden_dim": hidden_dim,
                "lr": lr,
                "bs": bs,
                "samples_per_experiment": samples_per_experiment,
                "val_samples_per_experiment": val_samples_per_experiment,
                "num_epochs": num_epochs,
                "rqs_K": rqs_K,
                "rqs_B": rqs_B,
                "model_type": model_type,
                "deepset_pool": deepset_pool,
                "normalization_info": normalization_info,
                "n_traj": n_traj,
            }
            if args_dict is not None:
                checkpoint["args"] = args_dict
            torch.save(checkpoint, save_path)
            logger.info(f"Saved checkpoint to {save_path}")

        if epoch % 5 == 0:
            # plot training/validation loss curves
            plt.figure()
            plt.plot(eval_epochs, train_loss_history, label="Train Loss")
            plt.plot(eval_epochs, val_loss_history, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss Curves")
            plt.legend()    
            plt_path = os.path.join(log_dir, f"loss_curve_Z{z_dim}.png")
            plt.savefig(plt_path)
            plt.close()


    logger.info("Training finished")

    


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="generate_dataset/data/trajectories.npy")
    parser.add_argument("--n_traj", type=int, default=10000, help="number of trajectories in the data folder")

    parser.add_argument(
        "--model_type",
        type=str,
        default="ARQS",
        choices=["ARQS"],
        help="RQS: coupling RQS (ConditionalFlowND), ARQS: nflows autoregressive RQS",
    )

    parser.add_argument("--epsilon", type=float, default=0.01, help="scale factor for std from KNN distance")

    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--bs", type=int, default=512, help="number of experiments per batch")
    parser.add_argument("--samples_per_experiment", type=int, default=600)
    parser.add_argument("--val_samples_per_experiment", type=int, default=600)

    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--rqs_K", type=int, default=16)
    parser.add_argument("--rqs_B", type=float, default=8, help="RQS tail bound in range [-B,B]")

    parser.add_argument("--z_dim", type=int, default=8, help="z_hat dimension")
    parser.add_argument("--hidden_dim", type=int, default=128)

    parser.add_argument("--deepset_pool", type=str, default="mean", choices=["mean", "sum", "max"])

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_epochs", type=int, default=200)

    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--save_dir", type=str, default="trained_nflow_gmm2/", help="Directory to save checkpoints")

    args = parser.parse_args()

    # Resolve device
    resolved_device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"

    # Build save directory name including epsilon and tag
    experiments_done = len(os.listdir(args.save_dir))
    save_dir = os.path.join(args.save_dir, f"exp{experiments_done + 1}", f"{args.model_type}",  f"epsilon{args.epsilon}", f"deepsetPool_{args.deepset_pool}", f"n_layers_{args.n_layers}", f"rqs_K_{args.rqs_K}", f"traj_{args.n_traj}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_model_Z{args.z_dim}.pth")



    train_conditional_flow_from_files(
        data_path=args.data_path,
        n_traj=args.n_traj,
        model_type=args.model_type,
        epsilon=args.epsilon,
        dim=args.dim,
        bs=args.bs,
        samples_per_experiment=args.samples_per_experiment,
        val_samples_per_experiment=args.val_samples_per_experiment,
        n_layers=args.n_layers,
        rqs_K=args.rqs_K,
        rqs_B=args.rqs_B,
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        deepset_pool=args.deepset_pool,
        lr=args.lr,
        num_epochs=args.num_epochs,
        seed=args.seed,
        device=resolved_device,
        save_path=save_path,
        args_dict=vars(args),
    )
