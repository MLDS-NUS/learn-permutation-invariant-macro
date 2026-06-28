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
try:
    from .torch_onsagernet import (
        ConservationMatrixMLP,
        DiffusionDiagonalConstant,
        DiffusionDiagonal,
        DissipationMatrixMLP,
        DriftMLP,
        DriftMLPSDE,
        OnsagerNetSDE,
        PotentialResMLP,
    )
except ImportError:  # fallback when running as a script
    from torch_onsagernet import (
        ConservationMatrixMLP,
        DiffusionDiagonalConstant,
        DiffusionDiagonal,
        DissipationMatrixMLP,
        DriftMLP,
        DriftMLPSDE,
        OnsagerNetSDE,
        PotentialResMLP,
    )


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)



def _load_mix_npz(npz_path: str, particle_type: int) -> np.ndarray:
    with np.load(npz_path) as data:
        macro_feature = data["macro_feature"]
        z_type1 = data["z_type1"]
        z_type2 = data["z_type2"]

    if macro_feature.ndim != 3 or macro_feature.shape[2] != 2:
        raise ValueError(
            f"macro_feature must have shape (n_traj, T, 2), got {macro_feature.shape}"
        )
    if z_type1.ndim != 3 or z_type2.ndim != 3:
        raise ValueError(
            "z_type1 and z_type2 must have shape (n_traj, T, z_dim)."
        )
    if z_type1.shape[:2] != macro_feature.shape[:2]:
        raise ValueError(
            "z_type1 must match macro_feature in (n_traj, T)."
        )
    if z_type2.shape[:2] != macro_feature.shape[:2]:
        raise ValueError(
            "z_type2 must match macro_feature in (n_traj, T)."
        )

    macro_idx = 0 if particle_type == 1 else 1
    macro_type = macro_feature[..., macro_idx : macro_idx + 1]
    # z_type = z_type1 if particle_type == 1 else z_type2
    z_type = np.concatenate([z_type1, z_type2], axis=-1) if particle_type == 1 else np.concatenate([z_type2, z_type1], axis=-1)
    return np.concatenate([macro_type, z_type], axis=-1)


def load_npy_data(data_path: str, particle_type: int) -> tuple[np.ndarray, np.ndarray]:
    if os.path.isdir(data_path):
        npz_path = os.path.join(data_path, "macro_and_Z_types.npz")
    else:
        npz_path = data_path

    if not npz_path.endswith(".npz"):
        raise ValueError(f"Expected a single .npz file, got {npz_path}")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing npz file: {npz_path}")

    data = _load_mix_npz(npz_path, particle_type)
    print(f"Loaded data shape: {data.shape}")
    n_traj = data.shape[0]
    split_idx = int(0.8 * n_traj)
    if split_idx <= 0 or split_idx >= n_traj:
        raise ValueError(
            f"Need at least 2 trajectories to split, got {n_traj}."
        )
    return data[:split_idx], data[split_idx:]


def compute_min_max(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    min_val = values.min(axis=(0, 1), keepdims=True)
    max_val = values.max(axis=(0, 1), keepdims=True)
    return min_val, max_val


def normalize_to_unit_range(
    values: np.ndarray,
    min_val: np.ndarray,
    max_val: np.ndarray,
) -> np.ndarray:
    scale = max_val - min_val
    scale[scale == 0] = 1.0
    return 2.0 * (values - min_val) / scale - 1.0




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
    particle_type: int,
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

    # train_data_np, val_data_np = load_groundTruth_Z(data_path)
    train_data_np, val_data_np = load_npy_data(data_path, particle_type)

    
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

    print("Normalizing inputs using train split min/max.")
    norm_min, norm_max = compute_min_max(train_data_np)
    train_data_np = normalize_to_unit_range(train_data_np, norm_min, norm_max)
    val_data_np = normalize_to_unit_range(val_data_np, norm_min, norm_max)

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

    def _checkpoint_payload() -> dict:
        return {
            "state_dict": model.state_dict(),
            "norm_min": norm_min,
            "norm_max": norm_max,
        }

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
                torch.save(_checkpoint_payload(), os.path.join(output_dir, "best_model.pt"))

            if checkpoint_every and epoch % checkpoint_every == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save(_checkpoint_payload(), ckpt_path)

            log_file.write(f"{epoch},{train_loss:.6f},{val_loss:.6f}\n")
            log_file.flush()

            print(
                f"Epoch {epoch:4d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
            )

    torch.save(_checkpoint_payload(), os.path.join(output_dir, "model.pt"))

    with open(os.path.join(output_dir, "loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    config_out = os.path.join(output_dir, "config.yaml")
    if not os.path.exists(config_out):
        with open(config_out, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "config", "mixing_dynamics.yaml")
    parser = argparse.ArgumentParser(
        description="PyTorch SDE training for polymer dynamics (closure variables only)."
    )
    parser.add_argument(
        "--data",
        default="../trained_nflow/exp1/dynamics_data_Z1/",
        help="Path to a macro_and_Z_types.npz file or its parent directory.",
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to the mixing_dynamics.yaml config.",
    )
        
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Torch device string, e.g. 0",
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=["float64"],
        help="Floating point precision to use.",
    )
    parser.add_argument(
        "--particle_type",
        type=str,
        default="type1",
        help="Which particle type to learn: type1/type2 (also accepts 1/2).",
    )
    args = parser.parse_args()

    particle_type = args.particle_type.strip().lower()
    if particle_type in ("1", "type1"):
        particle_type_value = 1
    elif particle_type in ("2", "type2"):
        particle_type_value = 2
    else:
        raise ValueError(
            f"particle_type must be 1/type1 or 2/type2, got {args.particle_type}"
        )
    config = load_config(args.config)
    diffusion_type = config["model"]["diffusion"].get("type", "unknown").lower()
    assert diffusion_type in ["constant", "diagonal"]
    
    output_dir = os.path.join(
        args.data,
        f"trained_sde_{args.particle_type}",
        diffusion_type,
    )
    print(f"Output directory: {output_dir}")


    run_training(
        data_path=args.data,
        config_path=args.config,
        output_dir=output_dir,
        device=args.device,
        dtype=args.dtype,
        particle_type=particle_type_value,
    )


if __name__ == "__main__":
    main()
