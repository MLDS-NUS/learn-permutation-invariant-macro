import argparse
import math
import os

import numpy as np
import torch

from learn_mixing_dynamics import build_model, load_config, normalize_to_unit_range
from mmd import compute_mmd_gpu


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
        raise ValueError("z_type1 and z_type2 must have shape (n_traj, T, z_dim).")
    if z_type1.shape[:2] != macro_feature.shape[:2]:
        raise ValueError("z_type1 must match macro_feature in (n_traj, T).")
    if z_type2.shape[:2] != macro_feature.shape[:2]:
        raise ValueError("z_type2 must match macro_feature in (n_traj, T).")

    macro_idx = 0 if particle_type == 1 else 1
    macro_type = macro_feature[..., macro_idx : macro_idx + 1]
    z_type = (
        np.concatenate([z_type1, z_type2], axis=-1)
        if particle_type == 1
        else np.concatenate([z_type2, z_type1], axis=-1)
    )
    return np.concatenate([macro_type, z_type], axis=-1)


def load_mix_test(data_path: str, particle_type: int) -> tuple[np.ndarray | None, np.ndarray]:
    if os.path.isdir(data_path):
        npz_path = os.path.join(data_path, "macro_and_Z_types.npz")
    else:
        npz_path = data_path

    if not npz_path.endswith(".npz"):
        raise ValueError(f"Expected a single .npz file, got {npz_path}")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing npz file: {npz_path}")

    data = _load_mix_npz(npz_path, particle_type)
    return None, data


def _extract_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint.get("norm_min"), checkpoint.get(
            "norm_max"
        )
    return checkpoint, None, None


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


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


def _denormalize_macro(values: np.ndarray, macro_min: np.ndarray, macro_max: np.ndarray) -> np.ndarray:
    macro_min_val = float(np.squeeze(macro_min))
    macro_max_val = float(np.squeeze(macro_max))
    scale = 0.5 * (macro_max_val - macro_min_val)
    return (values + 1.0) * scale + macro_min_val


def evaluate_type(
    particle_type_value: int,
    particle_label: str,
    data_path: str,
    config: dict,
    num_samples: int | None,
    device: torch.device,
    model_dir: str,
) -> dict:
    _, val_np = load_mix_test(data_path, particle_type_value)
    reduced_dim = config["reduced_dim"]
    if val_np.shape[-1] != reduced_dim:
        raise ValueError(
            f"{particle_label}: expected last dimension {reduced_dim}, got {val_np.shape[-1]}"
        )

    model_path = resolve_model_path(model_dir)
    model = build_model(config, reduced_dim, device=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict, norm_min, norm_max = _extract_checkpoint_state(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    norm_min = _to_numpy(norm_min)
    norm_max = _to_numpy(norm_max)
    if norm_min is None or norm_max is None:
        raise ValueError(f"Checkpoint {model_path} is missing norm_min/norm_max.")
    val_np = normalize_to_unit_range(val_np, norm_min, norm_max)
    macro_min = norm_min[..., 0]
    macro_max = norm_max[..., 0]

    steps = val_np.shape[1]
    if num_samples is None or num_samples <= 0:
        num_samples = val_np.shape[0]
    if num_samples > val_np.shape[0]:
        raise ValueError(
            f"{particle_label}: num_samples {num_samples} exceeds available {val_np.shape[0]}."
        )

    val_np = val_np[:num_samples]
    x0 = torch.as_tensor(
        val_np[:, 0],
        device=device,
        dtype=torch.get_default_dtype(),
    )
    dt_value = float(config["dt"])
    with torch.no_grad():
        sim = simulate_batch(model, x0, steps, dt_value)
    sim_np = sim.cpu().numpy()

    return {
        "label": particle_label,
        "steps": steps,
        "num_samples": num_samples,
        "pred_macro": _denormalize_macro(sim_np[:, :, :1], macro_min, macro_max),
        "true_macro": _denormalize_macro(val_np[:, :, :1], macro_min, macro_max),
    }


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "config", "mixing_dynamics.yaml")
    parser = argparse.ArgumentParser(
        description="Compute MMD between predicted and ground-truth macro trajectories."
    )
    parser.add_argument(
        "--data",
        default="../trained_nflow/exp3/dynamics_data_Z1/macro_and_Z_types_diffN_test.npz",
        help="Path to a macro_and_Z_types.npz file or its parent directory.",
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to the SDE config used for training.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of trajectories to evaluate (default: all in the test file).",
    )
    parser.add_argument(
        "--model_dir_type1",
        default=None,
        help="Directory with best_model.pt for type1.",
    )
    parser.add_argument(
        "--model_dir_type2",
        default=None,
        help="Directory with best_model.pt for type2.",
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

    data_root = args.data if os.path.isdir(args.data) else os.path.dirname(args.data)
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )

    config = load_config(args.config)
    diffusion_type = config["model"]["diffusion"].get("type", "unknown").lower()

    model_dir_type1 = args.model_dir_type1
    if model_dir_type1 is None:
        model_dir_type1 = os.path.join(
            data_root, "trained_sde_type1", diffusion_type
        )
    model_dir_type2 = args.model_dir_type2
    if model_dir_type2 is None:
        model_dir_type2 = os.path.join(
            data_root, "trained_sde_type2", diffusion_type
        )

    type1 = evaluate_type(
        particle_type_value=1,
        particle_label="type1",
        data_path=args.data,
        config=config,
        num_samples=args.num_samples,
        device=device,
        model_dir=model_dir_type1,
    )
    type2 = evaluate_type(
        particle_type_value=2,
        particle_label="type2",
        data_path=args.data,
        config=config,
        num_samples=args.num_samples,
        device=device,
        model_dir=model_dir_type2,
    )

    if type1["steps"] != type2["steps"]:
        raise ValueError("type1 and type2 have different trajectory lengths.")
    if type1["num_samples"] != type2["num_samples"]:
        raise ValueError("type1 and type2 have different trajectory counts.")

    pred = np.concatenate([type1["pred_macro"], type2["pred_macro"]], axis=-1)
    true = np.concatenate([type1["true_macro"], type2["true_macro"]], axis=-1)

    pred_t = torch.as_tensor(
        pred, device=device, dtype=torch.get_default_dtype()
    )
    true_t = torch.as_tensor(
        true, device=device, dtype=torch.get_default_dtype()
    )
    mmd_per_timestep = compute_mmd_gpu(pred_t, true_t)
    mean_mmd = float(mmd_per_timestep.mean())

    print("MMD per timestep shape:")
    print(mmd_per_timestep.shape)
    print(f"Mean MMD: {mean_mmd:.6e}")


if __name__ == "__main__":
    main()
