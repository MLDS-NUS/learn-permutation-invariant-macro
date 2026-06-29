import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from learn_polymer_dynamics import build_model, load_config


def denormalize_from_unit(values, min_val, max_val):
    scale = 0.5 * (max_val - min_val)
    return (values + 1.0) * scale + min_val


def load_polymer_test(
    data_path: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if not data_path.endswith(".npz"):
        raise ValueError(f"Expected a single .npz file, got {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing npz file: {data_path}")

    test_data = np.load(data_path)
    test_x_span_norm = test_data["x_span_norm"]
    test_z_norm = test_data["z_norm"]
    test_x_span_raw = test_data["x_span_raw"] if "x_span_raw" in test_data else None
    if "x_span_min" not in test_data or "x_span_max" not in test_data:
        raise ValueError("x_span_min/x_span_max are required to plot macro results.")
    x_span_min = float(test_data["x_span_min"])
    x_span_max = float(test_data["x_span_max"])

    assert test_x_span_norm.ndim == 2
    assert test_z_norm.ndim == 3
    start_idx = 0
    if test_x_span_raw is None:
        test_x_span_raw = denormalize_from_unit(
            test_x_span_norm, x_span_min, x_span_max
        )
    data = np.concatenate(
        [test_x_span_norm[:, start_idx:, np.newaxis], test_z_norm[:, start_idx:, :]],
        axis=-1,
    )
    return (
        test_x_span_raw[:, start_idx:],
        data,
        x_span_min,
        x_span_max,
    )


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


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_split_data(
    data_root: str,
    split: str,
    reduced_dim: int,
    num_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    input_file = os.path.join(data_root, f"{split}_Z_data.npz")
    x_span_raw, val_np, x_span_min, x_span_max = load_polymer_test(input_file)

    if val_np.shape[-1] != reduced_dim:
        raise ValueError(
            f"{split}: expected last dimension {reduced_dim}, got {val_np.shape[-1]}"
        )

    if num_samples is not None and num_samples > 0:
        if num_samples > val_np.shape[0]:
            raise ValueError(
                f"{split}: num_samples {num_samples} exceeds available {val_np.shape[0]}."
            )
        val_np = val_np[:num_samples]
        x_span_raw = x_span_raw[:num_samples]

    return x_span_raw, val_np, x_span_min, x_span_max


def compute_stats(
    model,
    val_np: np.ndarray,
    x_span_raw: np.ndarray,
    x_span_min: float,
    x_span_max: float,
    dt_value: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = val_np.shape[1]
    x0 = torch.as_tensor(val_np[:, 0], device=device, dtype=torch.get_default_dtype())
    with torch.no_grad():
        sim = simulate_batch(model, x0, steps, dt_value)

    x_span_min_t = torch.as_tensor(
        x_span_min, device=device, dtype=torch.get_default_dtype()
    )
    x_span_max_t = torch.as_tensor(
        x_span_max, device=device, dtype=torch.get_default_dtype()
    )
    pred_x_span = denormalize_from_unit(sim[:, :, 0], x_span_min_t, x_span_max_t)
    true_x_span = torch.as_tensor(
        x_span_raw, device=device, dtype=torch.get_default_dtype()
    )

    sim_mean = pred_x_span.mean(dim=0).cpu().numpy()
    sim_std = pred_x_span.std(dim=0).cpu().numpy()
    val_mean = true_x_span.mean(dim=0).cpu().numpy()
    val_std = true_x_span.std(dim=0).cpu().numpy()
    return sim_mean, sim_std, val_mean, val_std


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "config", "polymer_dynamics.yaml")
    parser = argparse.ArgumentParser(
        description="Plot macro x_span mean/std for test splits."
    )
    parser.add_argument(
        "--data_dir",
        default="../evaluation_nflow_pixels/Z2/macro_data_Z2",
        help="Path to a directory containing *_Z_data.npz files.",
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to the SDE config used for training.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of trajectories to evaluate (default: 100).",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Directory with best_model.pt.",
    )
    parser.add_argument(
        "--output_fig_name",
        type=str,
        default="macro_x_span.pdf",
        help="Output filename for the comparison figure.",
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

    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError("dtype must be float64")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    assert os.path.isdir(args.data_dir), f"Data directory not found: {args.data_dir}"
    data_root = args.data_dir
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )

    config = load_config(args.config)
    reduced_dim = config["reduced_dim"]

    diffusion_type = config["model"]["diffusion"].get("type", "unknown").lower()
    drift_cfg = config["model"].get("drift", {})
    drift_type = drift_cfg.get("type", "onsager").lower()
    model_dir = args.model_dir
    if model_dir is None:
        model_dir = os.path.join(
            args.data_dir,
            "train_sde",
            f"drift_{drift_type}",
            f"diffusion_{diffusion_type}",
        )
    print(f"Loading model from directory: {model_dir}")
    model_path = resolve_model_path(model_dir)

    model = build_model(config, reduced_dim, device=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(_extract_state_dict(checkpoint))
    model.eval()

    dt_value = float(config["dt"])

    output_dir = os.path.join(
        data_root, "evaluate_sde", f"drift_{drift_type}", f"diffusion_{diffusion_type}"
    )
    os.makedirs(output_dir, exist_ok=True)

    splits = ["test_fast", "test_medium", "test_slow"]
    title_map = {
        "test_fast": "test fast",
        "test_medium": "test medium",
        "test_slow": "test slow",
    }
    fig, axes = plt.subplots(
        1,
        len(splits),
        figsize=(3.25, 1.6),
        sharey=True,
    )
    axes = np.array(axes).reshape(1, len(splits))[0]

    for idx, split in enumerate(splits):
        x_span_raw, val_np, x_span_min, x_span_max = load_split_data(
            data_root, split, reduced_dim, args.num_samples
        )
        sim_mean, sim_std, val_mean, val_std = compute_stats(
            model,
            val_np,
            x_span_raw,
            x_span_min,
            x_span_max,
            dt_value,
            device,
        )
        steps = val_np.shape[1]
        t = np.arange(steps)
        ax = axes[idx]
        
        ax.fill_between(
            t,
            val_mean - val_std,
            val_mean + val_std,
            color="black",
            alpha=0.2,
            label="true std" if idx == 0 else None,
        )
        
        ax.fill_between(
            t,
            sim_mean - sim_std,
            sim_mean + sim_std,
            color="#1f77b4",
            alpha=0.2,
            label="ViT std" if idx == 0 else None,
        )
        ax.plot(t, val_mean, label="true mean", color="black", linewidth=0.8)
        ax.plot(t, sim_mean, label="ViT mean", color="#1f77b4", linewidth=0.8)
        ax.set_title(title_map.get(split, split), fontsize=6)
        ax.tick_params(axis="both", which="both", labelsize=6, width=0.6, length=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

    axes[0].set_ylabel("Stretching Length", fontsize=6)

    if len(axes) >= 2:
        axes[1].set_xlabel("Time Step", fontsize=6)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels) if len(labels) > 0 else 1,
        fontsize=6,
        frameon=True,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.8,
        labelspacing=0.2,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), pad=0.15)
    fig.subplots_adjust(wspace=0.15)

    out_path = os.path.join(output_dir, args.output_fig_name)
    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        bbox_extra_artists=(legend,),
    )
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
