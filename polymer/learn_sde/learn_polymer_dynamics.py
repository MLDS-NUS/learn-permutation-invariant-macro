import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
import pickle as pkl
from torch_onsagernet import (
    ConservationMatrixMLP,
    DiffusionDiagonalConstant,
    DiffusionDiagonal,
    DiffusionFull,
    DissipationMatrixMLP,
    DriftMLP,
    DriftMLPSDE,
    OnsagerNetSDE,
    PotentialResMLP,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)





def load_npy_data(data_path: str) -> tuple[np.ndarray, np.ndarray]:
    train_npz_file = os.path.join(data_path, "train_Z_data.npz")
    val_npz_file = os.path.join(data_path, "valid_Z_data.npz")
    if not os.path.exists(train_npz_file):
        raise FileNotFoundError(f"Training data file not found: {train_npz_file}")
    if not os.path.exists(val_npz_file):
        raise FileNotFoundError(f"Validation data file not found: {val_npz_file}")

    train_data = np.load(train_npz_file)
    valid_data = np.load(val_npz_file)

    train_x_span_norm = train_data["x_span_norm"]
    train_z_norm = train_data["z_norm"]
    val_x_span_norm = valid_data["x_span_norm"]
    val_z_norm = valid_data["z_norm"]

    assert train_x_span_norm.ndim == val_x_span_norm.ndim == 2
    assert train_z_norm.ndim == val_z_norm.ndim == 3
    start_idx = 0
    train_data = np.concatenate(
        [train_x_span_norm[:, start_idx:, np.newaxis], train_z_norm[:, start_idx:, :]], axis=-1
    )
    val_data = np.concatenate(
        [val_x_span_norm[:, start_idx:, np.newaxis], val_z_norm[:, start_idx:, :]], axis=-1
    )
    # print(f"Original train data shape: {train_data.shape}")
    # print(f"Original val data shape: {val_data.shape}")
    # exit()

    test_part = os.path.join(data_path, "test_slow_Z_data.npz")
    test_data = np.load(test_part)
    test_x_span_norm = test_data["x_span_norm"]
    test_z_norm = test_data["z_norm"]
    test_data = np.concatenate(
        [test_x_span_norm[:, start_idx:, np.newaxis], test_z_norm[:, start_idx:, :]], axis=-1
    )
    # train_data = np.concatenate([train_data, test_data[0:50]], axis=0)

    # new_train_data = np.concatenate([train_data, val_data[0:80].copy()], axis=0)
    # new_val_data = val_data[80:].copy()
    # train_data = new_train_data
    # val_data = new_val_data

    return train_data, val_data



def build_model(
    config: dict,
    dim: int,
    device: torch.device,
) -> OnsagerNetSDE | DriftMLPSDE:
    pot_cfg = config["model"]["potential"]
    dis_cfg = config["model"]["dissipation"]
    con_cfg = config["model"]["conservation"]
    diff_cfg = config["model"]["diffusion"]
    diffusion_type = diff_cfg.get("type", "none").lower()
    if diffusion_type == "constant":
        diffusion = DiffusionDiagonalConstant(
            dim=dim,
            alpha=diff_cfg["alpha"],
        )
    elif diffusion_type == "diagonal":
        if "units" not in diff_cfg or "activation" not in diff_cfg:
            raise ValueError(
                "diffusion.type=diagonal requires diffusion.units and diffusion.activation."
            )
        diffusion = DiffusionDiagonal(
            dim=dim,
            units=diff_cfg["units"],
            activation=diff_cfg["activation"],
            alpha=diff_cfg["alpha"],
        )
    elif diffusion_type == "full":
        if "units" not in diff_cfg or "activation" not in diff_cfg:
            raise ValueError(
                "diffusion.type=full requires diffusion.units and diffusion.activation."
            )
        diffusion = DiffusionFull(
            dim=dim,
            units=diff_cfg["units"],
            activation=diff_cfg["activation"],
            alpha=diff_cfg["alpha"],
        )
    else:
        raise ValueError(f"Unknown diffusion.type '{diffusion_type}'")

    drift_cfg = config["model"].get("drift", {})
    drift_type = drift_cfg.get("type", "onsager").lower()
    if drift_type == "onsager":
        potential = PotentialResMLP(
            dim=dim,
            units=pot_cfg["units"],
            activation=pot_cfg["activation"],
            n_pot=pot_cfg["n_pot"],
            alpha=pot_cfg["alpha"],
            param_dim=0,
        )
        dissipation = DissipationMatrixMLP(
            dim=dim,
            units=dis_cfg["units"],
            activation=dis_cfg["activation"],
            alpha=dis_cfg["alpha"],
            is_bounded=True,
        )
        conservation = ConservationMatrixMLP(
            dim=dim,
            units=con_cfg["units"],
            activation=con_cfg["activation"],
            is_bounded=True,
        )
        model = OnsagerNetSDE(
            potential=potential,
            dissipation=dissipation,
            conservation=conservation,
            diffusion=diffusion,
        )
    elif drift_type == "mlp":
        if "units" not in drift_cfg or "activation" not in drift_cfg:
            raise ValueError(
                "model.drift.type=mlp requires model.drift.units and model.drift.activation."
            )
        drift = DriftMLP(
            dim=dim,
            units=drift_cfg["units"],
            activation=drift_cfg["activation"],
        )
        model = DriftMLPSDE(
            drift=drift,
            diffusion=diffusion,
        )
    else:
        raise ValueError(f"Unknown model.drift.type '{drift_type}'")

    return model.to(device=device)


def mle_loss(
    model: OnsagerNetSDE | DriftMLPSDE,
    x0: torch.Tensor,
    x1: torch.Tensor,
    dt: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    drift = model.drift(None, x0, create_graph=create_graph)
    sigma = model.diffusion(None, x0)

    dt = dt.view(-1, 1)
    data = (x1 - x0) / dt
    cov = (1.0 / dt).view(-1, 1, 1) * (sigma @ sigma.transpose(-1, -2))

    diff = data - drift
    chol = torch.linalg.cholesky(cov)
    solved = torch.cholesky_solve(diff.unsqueeze(-1), chol).squeeze(-1)
    maha = torch.sum(diff * solved, dim=1)
    logdet = 2.0 * torch.sum(
        torch.log(torch.diagonal(chol, dim1=1, dim2=2)), dim=1
    )
    log2pi = math.log(2.0 * math.pi)
    logpdf = -0.5 * (diff.shape[1] * log2pi + logdet + maha)
    return -logpdf.mean()


def run_training(
    data_path: str,
    config_path: str,
    output_dir: str | None,
    device: int | None,
    dtype: str,
) -> None:
    config = load_config(config_path)

    seed = config["model"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    if dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        raise ValueError("dtype must be float32 or float64")

    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    train_data_np, val_data_np = load_npy_data(data_path)

    
    if train_data_np.ndim != 3:
        raise ValueError("Input data must have shape (n_traj, T, dim)")

    train_traj_len = config["train"].get("train_traj_len")
    if train_traj_len is not None:
        train_data_np = train_data_np[:, :train_traj_len, :]

    reduced_dim = config["reduced_dim"]
    if train_data_np.shape[-1] != reduced_dim:
        raise ValueError(
            f"Expected last dimension {reduced_dim}, got {train_data_np.shape[-1]}"
        )

    train_np, val_np = train_data_np, val_data_np
    default_dtype = torch.get_default_dtype()
    train_tensor = torch.as_tensor(train_np, dtype=default_dtype)
    val_tensor = torch.as_tensor(val_np, dtype=default_dtype)

    batch_size = config["train"]["batch_size"]
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_model(config, reduced_dim, device=device)

    def _as_float(value, name):
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"{name} must be a scalar, got {value}")
            value = value[0]
        return float(value)

    lr = _as_float(config["train"]["opt"]["learning_rate"], "learning_rate")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rop = config["train"]["rop"]
    min_scale = _as_float(rop.get("min_scale", 0.0), "min_scale")
    min_lr = lr * min_scale
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=_as_float(rop["factor"], "factor"),
        patience=int(rop["patience"]),
        cooldown=int(rop["cooldown"]),
        threshold=_as_float(rop["rtol"], "rtol"),
        min_lr=min_lr,
    )

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "loss_log.csv")

    dt_value = float(config["dt"])
    num_epochs = config["train"]["num_epochs"]
    checkpoint_every = config["train"]["checkpoint_every"]

    best_val = float("inf")
    history = {"train": [], "val": []}

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("epoch,train_loss,val_loss\n")
        log_file.flush()
        for epoch in range(1, num_epochs + 1):
            model.train()
            train_losses = []
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(device=device)
                x0 = batch_x[:, :-1, :]
                x1 = batch_x[:, 1:, :]
                batch, steps, dim = x0.shape
                x0_flat = x0.reshape(-1, dim)
                x1_flat = x1.reshape(-1, dim)

                dt = torch.full(
                    (x0_flat.shape[0], 1),
                    dt_value,
                    device=device,
                    dtype=x0_flat.dtype,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = mle_loss(model, x0_flat, x1_flat, dt)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            history["train"].append(train_loss)

            model.eval()
            val_losses = []
            with torch.no_grad():
                for (batch_x,) in val_loader:
                    batch_x = batch_x.to(device=device)
                    x0 = batch_x[:, :-1, :]
                    x1 = batch_x[:, 1:, :]
                    batch, steps, dim = x0.shape
                    x0_flat = x0.reshape(-1, dim)
                    x1_flat = x1.reshape(-1, dim)
                    dt = torch.full(
                        (x0_flat.shape[0], 1),
                        dt_value,
                        device=device,
                        dtype=x0_flat.dtype,
                    )
                    loss = mle_loss(model, x0_flat, x1_flat, dt, create_graph=False)
                    val_losses.append(loss.item())

            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            history["val"].append(val_loss)
            if val_losses:
                scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

            if checkpoint_every and epoch % checkpoint_every == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save(model.state_dict(), ckpt_path)

            log_file.write(f"{epoch},{train_loss:.6f},{val_loss:.6f}\n")
            log_file.flush()

            print(
                f"Epoch {epoch:4d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
            )

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    with open(os.path.join(output_dir, "loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    config_out = os.path.join(output_dir, "config.yaml")
    if not os.path.exists(config_out):
        with open(config_out, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "config", "polymer_dynamics.yaml")
    parser = argparse.ArgumentParser(
        description="PyTorch SDE training for polymer dynamics (closure variables only)."
    )
    parser.add_argument(
        "--data",
        default="../trained_nflow_images/Z2/macro_data_Z2/",
        help="Path to a macro_and_Z_types.npz file or its parent directory.",
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to the polymer_dynamics.yaml config.",
    )
        
    parser.add_argument(
        "--device",
        type=int,
        default=2,
        help="Torch device string, e.g. 0",
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=["float64"],
        help="Floating point precision to use.",
    )
    
    args = parser.parse_args()

    
    config = load_config(args.config)
    diffusion_type = config["model"]["diffusion"].get("type", "unknown").lower()
    assert diffusion_type in ["constant", "diagonal", "full"]
    

    drift_cfg = config["model"].get("drift", {})
    drift_type = drift_cfg.get("type", "onsager").lower()


    output_dir = os.path.join(
        args.data,
        "train_sde",
        f"drift_{drift_type}",
        f"diffusion_{diffusion_type}",
    )
    print(f"Output directory: {output_dir}")


    run_training(
        data_path=args.data,
        config_path=args.config,
        output_dir=output_dir,
        device=args.device,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
