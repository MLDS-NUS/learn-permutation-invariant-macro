import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from learn_mixing_dynamics import build_model, load_config, normalize_to_unit_range


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


def load_mix_split(data_path: str, particle_type: int) -> tuple[np.ndarray, np.ndarray]:
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


def load_mix_test(data_path: str, particle_type: int) -> tuple[np.ndarray, np.ndarray]:
    # load test data with the same initial state across trajectories
    if not data_path.endswith(".npz"):
        raise ValueError(f"Expected a single .npz file, got {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing npz file: {data_path}")

    data = _load_mix_npz(data_path, particle_type)
        
    return None, data


def resolve_model_path(model_dir: str) -> str:
    direct_best = os.path.join(model_dir, "best_model.pt")
    if os.path.exists(direct_best):
        return direct_best
    direct_final = os.path.join(model_dir, "model.pt")
    if os.path.exists(direct_final):
        return direct_final

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    subdirs = [
        os.path.join(model_dir, name)
        for name in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, name))
    ]
    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for subdir in subdirs:
        candidate = os.path.join(subdir, "best_model.pt")
        if os.path.exists(candidate):
            print("\nFound trained model at:")
            print(candidate)
            return candidate
        candidate = os.path.join(subdir, "model.pt")
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find best_model.pt or model.pt under {model_dir}."
    )


def simulate_batch(
    model,
    x0: torch.Tensor,
    steps: int,
    dt: float,
) -> torch.Tensor:
    traj = [x0]
    x = x0
    for _ in range(steps - 1):
        drift = model.drift(None, x, create_graph=False)
        sigma = model.diffusion(None, x)
        noise = torch.randn(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        x = x + drift * dt + (sigma @ noise.unsqueeze(-1)).squeeze(-1) * math.sqrt(dt)
        traj.append(x)
    return torch.stack(traj, dim=1)


def _extract_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint.get("norm_min"), checkpoint.get("norm_max")
    return checkpoint, None, None


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "config", "mixing_dynamics.yaml")
    parser = argparse.ArgumentParser(
        description="Evaluate trained SDE with Monte Carlo rollouts."
    )
    parser.add_argument(
        "--data",
        default="../trained_nflow/ARQS/epsilon0.01/deepsetPool_mean/n_layers_8/rqs_K_16/traj_200/dynamics_data_Z1/macro_and_Z_types_test_left.npz",
        help="Path to a macro_and_Z_types.npz file or its parent directory.",
    )
    parser.add_argument("--output_fig_name", type=str, default="mean_std_compare_left.png", help="Output filename for the comparison figure.")

    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to the SDE config used for training.",
    )
    parser.add_argument(
        "--particle_type",
        type=str,
        default="type1",
        help="Which particle type to evaluate: type1/type2 (also accepts 1/2).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of Monte Carlo rollouts.",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Directory with best_model.pt; defaults to trained_sde_{particle_type}.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Torch device index, e.g. 0.",
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=["float64"],
        help="Floating point precision to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for Monte Carlo rollouts.",
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

    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError("dtype must be float64")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    data_root = args.data if os.path.isdir(args.data) else os.path.dirname(args.data)
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )

    config = load_config(args.config)
    # Load validation data
    # _, val_np = load_mix_split(args.data, particle_type_value)
    _, val_np = load_mix_test(args.data, particle_type_value)

    reduced_dim = config["reduced_dim"]
    if val_np.shape[-1] != reduced_dim:
        raise ValueError(
            f"Expected last dimension {reduced_dim}, got {val_np.shape[-1]}"
        )

    model_dir = args.model_dir
    diffusion_type = config["model"]["diffusion"].get("type", "unknown").lower()
    if model_dir is None:
        model_dir = os.path.join(data_root, f"trained_sde_{args.particle_type}", diffusion_type)
    print(f"Loading model from directory: {model_dir}")    
    model_path = resolve_model_path(model_dir)

    model = build_model(config, reduced_dim, device=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict, norm_min, norm_max = _extract_checkpoint_state(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    dt_value = float(config["dt"])

    norm_min = _to_numpy(norm_min)
    norm_max = _to_numpy(norm_max)
    if norm_min is None or norm_max is None:
        raise ValueError(f"Checkpoint {model_path} is missing norm_min/norm_max.")
    val_np = normalize_to_unit_range(val_np, norm_min, norm_max)

    
    output_dir = os.path.join(data_root, f"evaluate_sde_{args.particle_type}", diffusion_type)
    
    print(f"Output directory: {output_dir}")


    os.makedirs(output_dir, exist_ok=True)

    steps = val_np.shape[1]
    x0 = torch.as_tensor(val_np[0, 0], device=device)
    x0 = x0.unsqueeze(0).repeat(args.num_samples, 1)

    with torch.no_grad():
        sim = simulate_batch(model, x0, steps, dt_value)
    sim_np = sim.cpu().numpy()
    sim_mean = sim_np.mean(axis=0)
    sim_std = sim_np.std(axis=0)

    val_mean = val_np.mean(axis=0)
    val_std = val_np.std(axis=0)

    t = np.arange(steps)
    n_dim = val_np.shape[2]
    n_cols = min(3, n_dim)
    n_rows = int(math.ceil(n_dim / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        sharex=True,
    )
    axes = np.array(axes).reshape(n_rows, n_cols)
    for dim in range(n_dim):
        ax = axes[dim // n_cols, dim % n_cols]
        ax.plot(
            t, val_mean[:, dim], label="val mean", color="black", linewidth=1.2
        )
        ax.fill_between(
            t,
            val_mean[:, dim] - val_std[:, dim],
            val_mean[:, dim] + val_std[:, dim],
            color="black",
            alpha=0.2,
            label="val std" if dim == 0 else None,
        )
        ax.plot(t, sim_mean[:, dim], label="sim mean", color="#1f77b4")
        ax.fill_between(
            t,
            sim_mean[:, dim] - sim_std[:, dim],
            sim_mean[:, dim] + sim_std[:, dim],
            color="#1f77b4",
            alpha=0.2,
            label="sim std" if dim == 0 else None,
        )
        ax.set_ylabel(f"dim {dim}")
    for ax in axes[-1, :]:
        ax.set_xlabel("time step")
    axes[0, 0].legend(loc="best")
    fig.suptitle(
        f"SDE rollouts vs. validation mean/std ({args.particle_type})"
    )
    fig.tight_layout()

    out_path = os.path.join(output_dir, args.output_fig_name)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
