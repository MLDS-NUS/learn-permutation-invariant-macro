import os
import logging
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm

import matplotlib.pyplot as plt

from utils.utils import set_seed

# Your existing model components
from models import (
    AnchorGaussianMixtureND,
    SetEncoderND,
)

# nflows: autoregressive RQS
from nflows import transforms as nf_transforms
from nflows import distributions as nf_distributions
from nflows import flows as nf_flows


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _should_record(time_step: int) -> bool:
    i = time_step + 1
    if i <= 300:
        return True
    elif i % 20 == 0:
        return True
    else:
        return False
    
    
        
def _load_input_arrays(data_path: str):
    data = np.load(data_path, allow_pickle=True)
    input_data = data["trajectories"]
    types = data["types"]
    data.close()

    return input_data, types

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
    If types are provided, they are expanded per timestep and returned alongside anchors.
    """
    
    input_data, types = _load_input_arrays(data_path)
    assert input_data.shape[0] >= n_traj, f"data has {input_data.shape[0]} trajectories, but n_traj={n_traj}"
    input_data = input_data[:n_traj]  # (E, T, M, D)
    if types is not None:
        if types.ndim != 2:
            raise ValueError(f"types must be 2D (num_traj, N), got shape {types.shape}")
        if types.shape[0] < n_traj:
            raise ValueError(f"types has {types.shape[0]} trajectories, but n_traj={n_traj}")
        types = types[:n_traj]
        if types.shape[1] != input_data.shape[2]:
            raise ValueError(
                f"types has N={types.shape[1]}, but data has N={input_data.shape[2]}"
            )
        unique_types = np.unique(types)
        if not np.all(np.isin(unique_types, [1, 2])):
            raise ValueError(f"types must contain only 1 and 2, got {unique_types}")
    print(f"input_data shape: {input_data.shape}")
    if types is not None:
        print(f"types shape: {types.shape}")

    # Save time by selecting timesteps according to _should_record
    selected_timesteps = [t for t in range(input_data.shape[1]) if _should_record(t)]
    input_data = input_data[:, selected_timesteps, :, :]  # (E, T_sel, M, D)
    print(f"Selected {len(selected_timesteps)} timesteps: {selected_timesteps}")
    print(f"input_data shape after timestep selection: {input_data.shape}")
        
    
    # normalize data to [-1, 1] in each dimension
    data_min = input_data.min(axis=(0, 1, 2), keepdims=True)
    data_max = input_data.max(axis=(0, 1, 2), keepdims=True)
    scale = data_max - data_min
    scale[scale == 0] = 1.0
    norm_input_data = 2.0 * (input_data - data_min) / scale - 1.0
    
    normalization_info = {
        "min": data_min,
        "max": data_max,
    }
    print(f"Data normalization info: {normalization_info}")
    

    

    train_num = int(0.8 * norm_input_data.shape[0])
    train_pos = norm_input_data[:train_num]   # (E_train, T, M, D)
    val_pos = norm_input_data[train_num:]     # (E_val, T, M, D)
    if types is not None:
        train_types = types[:train_num]
        val_types = types[train_num:]
    else:
        train_types = None
        val_types = None
    
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
    if train_types is not None:
        train_types = np.repeat(train_types[:, None, :], train_pos.shape[1], axis=1)
        val_types = np.repeat(val_types[:, None, :], val_pos.shape[1], axis=1)
        train_types = torch.from_numpy(train_types.reshape(-1, train_types.shape[2])).long()
        val_types = torch.from_numpy(val_types.reshape(-1, val_types.shape[2])).long()

    assert train_anchors.dim() == 3
    assert val_anchors.dim() == 3

    # Wrap in datasets/loaders; each item is a single experiment's anchor set: (M,D)
    if train_types is None:
        train_ds = TensorDataset(train_anchors)
        val_ds = TensorDataset(val_anchors)
    else:
        train_ds = TensorDataset(train_anchors, train_types)
        val_ds = TensorDataset(val_anchors, val_types)
    print(f"train_anchors shape: {train_anchors.shape}, val_anchors shape: {val_anchors.shape}")
    if train_types is not None:
        print(f"train_types shape: {train_types.shape}, val_types shape: {val_types.shape}")

    train_loader = DataLoader(train_ds, batch_size=bs, pin_memory=True, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, pin_memory=True, shuffle=True, drop_last=False)

    dim = train_pos.shape[3]
    n_anchors = train_pos.shape[2]

    return train_loader, val_loader, dim, n_anchors, normalization_info, (types is not None)


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
        
        # DeepSet encoder to get context per experiment: (E,z_dim)
        z_ctx = self.set_encoder(anchors)  # (E,z_dim)
        return self.log_prob_with_context(x, z_ctx)

    def log_prob_with_context(self, x: torch.Tensor, z_ctx: torch.Tensor) -> torch.Tensor:
        """
        x:     (E,B,D) or (B,D)
        z_ctx: (E,z_dim) or (z_dim,)
        returns: (E,B)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if z_ctx.dim() == 1:
            z_ctx = z_ctx.unsqueeze(0)

        E, B, D = x.shape
        assert D == self.dim
        if z_ctx.shape[0] != E:
            raise ValueError(f"context batch {z_ctx.shape[0]} does not match x batch {E}")

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

        # # Keep everything on the model device
        # device = next(self.parameters()).device
        # anchors = anchors.to(device)

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


def _sample_masked_anchor_gmm(
    anchors: torch.Tensor,
    mask: torch.Tensor,
    num_samples: int,
    epsilon: float,
) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Encountered a trajectory without any particles of the requested type.")
    probs = mask.float() / counts.unsqueeze(1)
    comp_ids = torch.multinomial(probs, num_samples, replacement=True)
    exp_ids = torch.arange(anchors.shape[0], device=anchors.device).unsqueeze(1)
    exp_ids = exp_ids.expand(anchors.shape[0], num_samples)
    selected = anchors[exp_ids, comp_ids]
    eps = torch.randn_like(selected)
    return selected + epsilon * eps


def train_conditional_flow(
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
    - DeepSet encoder is applied separately to each particle type.

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
    train_loader, val_loader, inferred_dim, n_anchors, normalization_info, has_types = make_dataloaders(
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
    train_batch_fraction = 0.1
    val_batch_fraction = 0.2
    rng = np.random.default_rng()
    # ----------------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------------
    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        train_losses = []
        train_loader_epoch = _make_subset_loader(train_loader, train_batch_fraction, rng)
        for batch in train_loader_epoch:
            if has_types:
                anchors_batch, types_batch = batch
                anchors_batch = anchors_batch.to(device, non_blocking=True)  # (E_batch, N, D)
                types_batch = types_batch.to(device, non_blocking=True)      # (E_batch, N)
                mask_type1 = types_batch == 1
                mask_type2 = types_batch == 2

                z_type1 = model.set_encoder(anchors_batch, mask=mask_type1)
                z_type2 = model.set_encoder(anchors_batch, mask=mask_type2)

                x_type1 = _sample_masked_anchor_gmm(
                    anchors_batch, mask_type1, samples_per_experiment, epsilon
                )
                x_type2 = _sample_masked_anchor_gmm(
                    anchors_batch, mask_type2, samples_per_experiment, epsilon
                )
                # print(f"x_type1 shape: {x_type1.shape}, x_type2 shape: {x_type2.shape}"); exit()

                log_p_type1 = model.log_prob_with_context(x_type1, z_type1)
                log_p_type2 = model.log_prob_with_context(x_type2, z_type2)
                loss = -0.5 * (log_p_type1.mean() + log_p_type2.mean())
            else:
                anchors_batch = batch[0]
                anchors_batch = anchors_batch.to(device, non_blocking=True)  # (E_batch, M, D)

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
        with torch.no_grad():
            val_loader_epoch = _make_subset_loader(val_loader, val_batch_fraction, rng)
            for batch in val_loader_epoch:
                if has_types:
                    anchors_batch_val, types_batch_val = batch
                    anchors_batch_val = anchors_batch_val.to(device, non_blocking=True)
                    types_batch_val = types_batch_val.to(device, non_blocking=True)
                    mask_type1 = types_batch_val == 1
                    mask_type2 = types_batch_val == 2

                    z_type1 = model.set_encoder(anchors_batch_val, mask=mask_type1)
                    z_type2 = model.set_encoder(anchors_batch_val, mask=mask_type2)

                    x_type1 = _sample_masked_anchor_gmm(
                        anchors_batch_val, mask_type1, val_samples_per_experiment, epsilon
                    )
                    x_type2 = _sample_masked_anchor_gmm(
                        anchors_batch_val, mask_type2, val_samples_per_experiment, epsilon
                    )

                    log_p_type1 = model.log_prob_with_context(x_type1, z_type1)
                    log_p_type2 = model.log_prob_with_context(x_type2, z_type2)

                    val_loss = -0.5 * (log_p_type1.mean() + log_p_type2.mean())
                else:
                    anchors_batch_val = batch[0]
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
                "has_types": has_types,
                "seed": seed,
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

    parser.add_argument("--data_path", type=str, default="generate_data/dataset/trajectories_large.npz")
    parser.add_argument("--n_traj", type=int, default=10000, help="number of trajectories in the data folder")

    parser.add_argument(
        "--model_type",
        type=str,
        default="ARQS",
        choices=["ARQS"],
        help="RQS: coupling RQS (ConditionalFlowND), ARQS: nflows autoregressive RQS",
    )

    parser.add_argument("--epsilon", type=float, default=0.01, help="scale factor for std from KNN distance")

    parser.add_argument("--dim", type=int, default=2, help="data dimension (2 for 2D)")
    parser.add_argument("--bs", type=int, default=512, help="number of experiments per batch")
    parser.add_argument("--samples_per_experiment", type=int, default=800)
    parser.add_argument("--val_samples_per_experiment", type=int, default=800)

    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--rqs_K", type=int, default=16)
    parser.add_argument("--rqs_B", type=float, default=5, help="RQS tail bound in range [-B,B]")

    parser.add_argument("--z_dim", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=128)

    parser.add_argument("--deepset_pool", type=str, default="mean", choices=["mean", "sum", "max"])

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_epochs", type=int, default=50)

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_dir", type=str, default="trained_nflow/", help="Directory to save checkpoints")

    args = parser.parse_args()

    # Resolve device
    resolved_device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"

    # Build save directory name including epsilon and tag
    os.makedirs(args.save_dir, exist_ok=True)
    experiments_done = len(os.listdir(args.save_dir))
    save_dir = os.path.join(args.save_dir, f"exp{experiments_done + 1}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_model_Z{args.z_dim}.pth")



    train_conditional_flow(
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
