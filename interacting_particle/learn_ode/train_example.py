import argparse
import os
from tqdm import tqdm
import numpy as np
import math
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
# this repo's dynamics model
from models import OnsagerNet_original
from models import drift_MLP
import random

# Set the random seed for reproduction
def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_datasets(data, train_frac=0.8):
    """
    data: torch.Tensor [num_tra, T, D]
    returns: train_loader, val_loader, T
    """
    num_tra, T, D = data.shape
    n_train = int(train_frac * num_tra)

    # split trajectories
    data_train = data[:n_train]          # [n_train, T, D]
    data_val   = data[n_train:]          # [n_val,   T, D]
    # data_train = data_train[:, 0:150]   # limit to first 80 steps for faster training
    # data_val   = data_val[:, 0:150]

    # build 1-step pairs: (z_t, z_{t+dt})
    z0_train = data_train[:, :-1, :]     # [n_train, T-1, D]
    z1_train = data_train[:, 1:,  :]     # [n_train, T-1, D]
    z0_val   = data_val[:,   :-1, :]
    z1_val   = data_val[:,   1:,  :]


    # flatten trajectories -> big batch of 1-step transitions
    n_dim = D
    print("n_dim: ", n_dim)

    z0_train_flat = z0_train.reshape(-1, n_dim)   # [N_train, D]
    z1_train_flat = z1_train.reshape(-1, n_dim)

    z0_val_flat = z0_val.reshape(-1, n_dim)
    z1_val_flat = z1_val.reshape(-1, n_dim)

    train_ds = TensorDataset(z0_train_flat, z1_train_flat)
    val_ds   = TensorDataset(z0_val_flat,   z1_val_flat)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)

    return train_loader, val_loader, T, n_dim


def train_ode(
    data_path,
    dt_scalar,
    n_epochs=200,
    lr=1e-3,
    train_frac=0.8,
    drift_model="OnsagerNet",
    save_path="best_drift_mlp.pth",
    weight_alpha=1.0,
    weight_eps=1e-6,
    seed=42,
    device=None,
):
    # ----- setup -----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    set_seed(seed)

    # load data: [num_tra, T, D]
    data_file = os.path.join(data_path, "macro_input", "macro_and_Z_over_time.npz")
    data_npz = np.load(data_file)
    macro = data_npz["macro"]
    Z = data_npz["Z"]
    if macro.ndim == 2:
        macro = macro[:, :, None]

    # data = data[..., 0:1] # only use first dimension for ODE learning
    print(f"macro shape: {macro.shape}, Z shape: {Z.shape}")
    num_tra = macro.shape[0]
    n_train = int(train_frac * num_tra)
    Z_train = Z[:n_train]
    Z_min = Z_train.min(axis=(0, 1))
    Z_max = Z_train.max(axis=(0, 1))
    Z = 2.0 * (Z - Z_min) / (Z_max - Z_min) - 1.0
    normalization_info = {
        "z_min": Z_min,
        "z_max": Z_max,
    }
    data = np.concatenate([macro, Z], axis=-1)
    print(f"data shape: {data.shape}")
    data = torch.from_numpy(data).float().to(device)  # [num_tra, T, D]


    train_loader, val_loader, T, n_dim = make_datasets(
        data, train_frac=train_frac
    )

    # ----- model -----
    if drift_model == "OnsagerNet":
        model = OnsagerNet_original(input_dim=n_dim).to(device)
    elif drift_model == "MLP":
        model = drift_MLP(input_dim=n_dim).to(device)
    else:
        raise ValueError(f"Unknown drift_model: {drift_model}")
    metric = torch.nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    # ----- training loop -----
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []

        for z0, z1 in train_loader:
            opt.zero_grad()
            dzdt = model(z0)   # [B, D]
            true_dzdt = (z1 - z0) / dt_scalar
            weights = (torch.abs(z1 - z0) + weight_eps).pow(weight_alpha)
            loss = (weights * (dzdt - true_dzdt).pow(2)).mean()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        # validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            for z0, z1 in val_loader:
                dzdt = model(z0)   # [B, D]
                true_dzdt = (z1 - z0) / dt_scalar
                weights = (torch.abs(z1 - z0) + weight_eps).pow(weight_alpha)
                # loss = (weights[..., 0:1] * (dzdt[..., 0:1] - true_dzdt[..., 0:1]).pow(2)).mean()
                loss = (weights * (dzdt - true_dzdt).pow(2)).mean()
                val_losses.append(loss.item())
            val_loss = float(np.mean(val_losses))

        print(f"Epoch {epoch:4d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_file = os.path.join(save_path, "best_drift_mlp.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "normalization_info": normalization_info,
                },
                save_file,
            )
            print(f"  -> New best model saved to {save_file}")
    

        # ----- simple trajectory test -----
        if epoch % 5 == 0:
            # Take the first validation trajectory and compare prediction vs truth
            with torch.no_grad():
                if data.shape[0] > 0:
                    # initial state of the first val trajectory
                    # z0_val_traj = data[int(train_frac * data.shape[0])]   # [T, D]
                    
                    z0_val_traj = data[0]   # [T, D] # first trajectory in the dataset

                    steps = z0_val_traj.shape[0]
                    z0 = z0_val_traj[0:1, :]                              # [1, D]

                    predict_tra = [z0]
                    for _ in range(steps-1):
                        x0 = predict_tra[-1]
                        dzdt = model(x0)  # [1, D]
                        x1 = x0 + dzdt * dt_scalar 
                        predict_tra.append(x1)

                    predict_tra = torch.cat(predict_tra, dim=0)  # [T, D]

                    # Save comparison plot for all dimensions in n_row x 3 grid
                    t = np.arange(steps) * dt_scalar

                    n_row = math.ceil(n_dim / 3)
                    fig, axes = plt.subplots(n_row, 3, figsize=(12, 4 * n_row))
                    axes = np.atleast_1d(axes).ravel()

                    for d in range(n_dim):
                        ax = axes[d]
                        true_d = z0_val_traj[:, d].detach().cpu().numpy()
                        pred_d = predict_tra[:, d].detach().cpu().numpy()
                        ax.plot(t, true_d, label=f"true dim{d}")
                        ax.plot(t, pred_d, label=f"pred dim{d}", linestyle="--")
                        ax.set_xlabel("time")
                        ax.set_ylabel(f"z[{d}]")
                        ax.legend()

                    # Hide any unused subplots
                    for i in range(n_row * 3):
                        if i >= n_dim:
                            axes[i].axis("off")

                    fig.tight_layout()
                    out_path = os.path.join(save_path, "true_vs_pred_all_dims.png")
                    fig.savefig(out_path, dpi=200)
                    plt.close(fig)
                    # print(f"Saved comparison plot to {out_path}")
            # plot the true and target dzdt for first graph
            with torch.no_grad():
                z0_val_traj = data[0]   # [T, D] # first trajectory in the dataset
                dzdt_true = (z0_val_traj[1:, :] - z0_val_traj[:-1, :]) / dt_scalar  # [T-1, D]
                z0_input = z0_val_traj[:-1, :]  # [T -1, D]
                dzdt_pred = model(z0_input)      # [T-1, D]
                # in n_row * 3 grid
                n_row = math.ceil(n_dim / 3)
                fig, axes = plt.subplots(n_row, 3, figsize=(12, 4 * n_row))
                axes = np.atleast_1d(axes).ravel()
                for d in range(n_dim):
                    ax = axes[d]
                    true_d = dzdt_true[:, d].detach().cpu().numpy()
                    pred_d = dzdt_pred[:, d].detach().cpu().numpy()
                    ax.plot(true_d, label=f"true dzdt dim{d}")
                    ax.plot(pred_d, label=f"pred dzdt dim{d}", linestyle="--")
                    ax.set_xlabel("time step")
                    ax.set_ylabel(f"dzdt[{d}]")
                    ax.legend()
                # Hide any unused subplots
                for i in range(n_row * 3):
                    if i >= n_dim:
                        axes[i].axis("off")
                fig.tight_layout()
                out_path = os.path.join(save_path, "true_vs_pred_dzdt_all_dims.png")
                fig.savefig(out_path, dpi=200)
                plt.close(fig)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../evaluation_nflow_gmm2/exp1/", help="Path to macro_input.")    
    parser.add_argument("--dt", type=float, default=0.002,
                        help="Time step between successive slices in the trajectories.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--weight_alpha", type=float, default=0.0,
                        help="Exponent for magnitude-based loss weighting.")
    parser.add_argument("--weight_eps", type=float, default=1e-6,
                        help="Epsilon added to |z1 - z0| before exponent.")
    parser.add_argument("--drift_model", type=str, default="MLP", choices=["MLP", "OnsagerNet"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
        
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(args.data_path, "learned_dynamics", args.drift_model)
    os.makedirs(save_path, exist_ok=True)
    train_ode(
        data_path=args.data_path,
        dt_scalar=args.dt,
        n_epochs=args.epochs,
        lr=args.lr,
        train_frac=args.train_frac,
        drift_model=args.drift_model,
        save_path=save_path,
        weight_alpha=args.weight_alpha,
        weight_eps=args.weight_eps,
        seed=args.seed,
        device=device,        
    )
