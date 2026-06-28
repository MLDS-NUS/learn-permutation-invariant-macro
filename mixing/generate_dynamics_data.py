import os
import argparse
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from models import SetEncoderND
from train_nflow_arqs import NFlowsConditionalARQS  # AR RQS wrapper from training script

MODE_PATHS = {
    "train": (
        os.path.join("generate_data", "dataset", "trajectories_large.npz"),
        os.path.join("generate_data", "dataset", "macro_feature_large.npy"),
        "macro_and_Z_types.npz",
    ),
    "in_dst_test": (
        os.path.join("generate_data", "dataset", "trajectories_inDistribution_test.npz"),
        os.path.join("generate_data", "dataset", "macro_feature_inDistribution_test.npy"),
        "macro_and_Z_types_test_inDistribution.npz",
    ),
    "test_left": (
        os.path.join("generate_data", "dataset", "trajectories_test_left.npz"),
        os.path.join("generate_data", "dataset", "macro_feature_test_left.npy"),
        "macro_and_Z_types_test_left.npz",
    ),
    "test_mid": (
        os.path.join("generate_data", "dataset", "trajectories_test_mid.npz"),
        os.path.join("generate_data", "dataset", "macro_feature_test_mid.npy"),
        "macro_and_Z_types_test_mid.npz",
    ),
    "test_right": (
        os.path.join("generate_data", "dataset", "trajectories_test_right.npz"),
        os.path.join("generate_data", "dataset", "macro_feature_test_right.npy"),
        "macro_and_Z_types_test_right.npz",
    ),
    "diff_dst_test": (
        os.path.join("generate_data", "dataset", "trajectories_diffInitPos_test.npz"),
        os.path.join("generate_data", "dataset", "macro_feature_diffInitPos_test.npy"),
        "macro_and_Z_types_diffInitPos_test.npz",
    ),
    "diff_N_test": (
        os.path.join("generate_data", "dataset", "trajectories_diffN_test.npz"),
        os.path.join("generate_data", "dataset", "macro_feature_diffN_test.npy"),
        "macro_and_Z_types_diffN_test.npz",
    ),
}


# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------
def _should_record(time_step: int) -> bool:
    # return (time_step + 1) <= 300
    return True


def _load_trajectories_from_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    trajectories = data["trajectories"]
    types = data["types"]
    data.close()
    return trajectories, types


def load_anchors_from_npz(
    data_path: str,
    time_step=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load anchor sets and particle types from the NPZ file used in training.

    Returns:
      anchors_all: (E, S, M, D) where S = number of selected time steps
      types:       (E, M) int array with values {1,2}
      time_idx:    (S,) numpy array of the resolved time indices (0-based)
    """
    anchors, types = _load_trajectories_from_npz(data_path)  # (E, T, M, D), (E, M)
    E, T, M, D = anchors.shape
    if types.shape != (E, M):
        raise ValueError(f"types must have shape {(E, M)}, got {types.shape}")

    if time_step is None:
        time_step = [i for i in range(T) if _should_record(i)]
        if not time_step:
            raise ValueError("No time steps selected by _should_record().")

    if isinstance(time_step, (int, np.integer)):
        time_idx = np.array([int(time_step)], dtype=int)
    else:
        time_idx = np.array(list(time_step), dtype=int)

    resolved = []
    for ts in time_idx:
        ts_res = T + ts if ts < 0 else ts
        if not (0 <= ts_res < T):
            raise IndexError(
                f"time_step {ts} (resolved to {ts_res}) is out of range [0, {T-1}]"
            )
        resolved.append(ts_res)
    time_idx = np.array(resolved, dtype=int)

    anchors_all = anchors[:, time_idx, :, :]  # (E, S, M, D)
    return anchors_all, types, time_idx


def _reshape_norm_values(values: np.ndarray, ndim: int) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    shape = (1,) * (ndim - 1) + (flat.shape[0],)
    return flat.reshape(shape)


def normalize_anchors(
    anchors: np.ndarray,
    normalization_info: dict,
) -> np.ndarray:
    min_val = _reshape_norm_values(normalization_info["min"], anchors.ndim)
    max_val = _reshape_norm_values(normalization_info["max"], anchors.ndim)
    scale = max_val - min_val
    scale[scale == 0] = 1.0
    return 2.0 * (anchors - min_val) / scale - 1.0

# ---------------------------------------------------------------------------
# Rebuild model + scale from checkpoint (RQS or ARQS)
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, dict, dict]:
    """
    Reconstruct the trained model (RQS or ARQS) from the checkpoint and
    also reconstruct the min/max normalization used during training.

    Returns:
      model:  nn.Module in eval mode
      ckpt:   checkpoint dict
      normalization_info: dict with min/max used for training normalization
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    dim       = ckpt["dim"]
    model_type = ckpt.get("model_type", "ARQS").upper()
    z_dim     = ckpt.get("z_dim", ckpt.get("args", {}).get("z_dim", 4))
    hidden    = ckpt.get("hidden_dim", ckpt.get("args", {}).get("hidden_dim", 256))
    n_layers  = ckpt.get("n_layers", ckpt.get("args", {}).get("n_layers", 8))
    rqs_K     = ckpt.get("rqs_K", ckpt.get("args", {}).get("rqs_K", 16))
    rqs_B     = ckpt.get("rqs_B", ckpt.get("args", {}).get("rqs_B", 5.0))
    deepset_pool = ckpt.get("deepset_pool", ckpt.get("args", {}).get("deepset_pool", "mean"))

    print(f"Checkpoint loaded from: {ckpt_path}")
    print(f"dim={dim}, z_dim={z_dim}, hidden_dim={hidden}, n_layers={n_layers}, "
          f"rqs_K={rqs_K}, rqs_B={rqs_B}, model_type={model_type}")

    # Shared DeepSet encoder
    set_encoder = SetEncoderND(
        in_dim=dim,
        hidden_dim=hidden,
        z_dim=z_dim,
        pool=deepset_pool,
    ).to(device)

    

    # nflows-based autoregressive RQS flow (same wrapper as training)
    model = NFlowsConditionalARQS(
        dim=dim,
        z_dim=z_dim,
        hidden_dim=hidden,
        n_layers=n_layers,
        K=rqs_K,
        B=rqs_B,
        set_encoder=set_encoder,
    ).to(device)


    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    normalization_info = ckpt.get("normalization_info", ckpt.get("args", {}).get("normalization_info", None))
    return model, ckpt, normalization_info


def get_deepset_z_over_time(
    model,
    anchors: np.ndarray,  # (E, T, N, D)
    types: np.ndarray,    # (E, N)
    particle_type: int,
    device,
) -> np.ndarray:
    # anchors_scaled = torch.from_numpy(anchors).float().to(device)  # (E, T, N, D)
    E, T, N, D = anchors.shape
    set_encoder = model.set_encoder
    set_encoder.eval()
    model.eval()

    Z_list = []
    with torch.no_grad():
        for e in tqdm(range(E), desc=f"Z over time for type {particle_type}"):
            mask_np = types[e] == particle_type
            if not np.any(mask_np):
                raise ValueError(f"No particles of type {particle_type} in experiment {e}.")
            mask_t = torch.from_numpy(mask_np).to(device)
            anchors_e = torch.from_numpy(anchors[e]).float().to(device).contiguous()  # (T, N, D)
            mask_e = mask_t.unsqueeze(0).expand(T, N)
            Z_e = set_encoder(anchors_e, mask=mask_e)  # (T, z_dim)
            Z_list.append(Z_e.cpu().numpy())
    Z = np.stack(Z_list, axis=0)  # (E, T, z_dim)
    return Z






# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained conditional NF (RQS/ARQS)")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Directory to checkpoint", # e.g., trained_nflow/ARQS/epsilon0.01/deepsetPool_mean/n_layers_8/rqs_K_16/traj_200/
    )
    parser.add_argument("--z_dim", type=int, default=1, help="output dimension of deepset")

    parser.add_argument(
        "--mode",
        type=str,
        choices=tuple(MODE_PATHS),
        default="train",
        help="Dataset preset to use for the default trajectory and macro-feature paths.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="NPZ path with 'trajectories' and 'types'. Overrides the path implied by --mode.",
    )
    parser.add_argument(
        "--macro_path",
        type=str,
        default=None,
        help="Path to macro_feature.npy (shape: n_traj, T, 2). Overrides the path implied by --mode.",
    )
    parser.add_argument(
        "--time_step",
        type=int,
        nargs="+",
        default=None,
        help="Optional time steps; default matches training selection.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output NPZ filename. Overrides the name implied by --mode.",
    )

    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()
    mode_data_path, mode_macro_path, mode_output_name = MODE_PATHS[args.mode]
    args.data_path = args.data_path or os.path.join(script_dir, mode_data_path)
    args.macro_path = args.macro_path or os.path.join(script_dir, mode_macro_path)
    args.output_name = args.output_name or mode_output_name

        

    ## load data
    # input_data = np.load(args.data_path)
    # print(f"input_data shape: {input_data.shape}")  # (E,T,N,D)

    

    ## load model

    # Load model + scale
    device = torch.device(
        f"cuda:{args.device}"
        if (args.device >= 0 and torch.cuda.is_available())
        else "cpu"
    )
    ckpt_path = os.path.join(
        args.base_path,
        f"best_model_Z{args.z_dim}.pth",
    )
    model, ckpt, normalization_info = build_model_from_checkpoint(ckpt_path, device)
    if normalization_info is None:
        raise ValueError("Checkpoint is missing normalization_info.")
    print(f"normalization_info: {normalization_info}")

    output_root = os.path.join(args.base_path, f"dynamics_data_Z{args.z_dim}")
    os.makedirs(output_root, exist_ok=True)

    anchors, types, time_idx = load_anchors_from_npz(
        data_path=args.data_path,
        time_step=args.time_step,
    )
    print(f"Input anchors shape: {anchors.shape}, types shape: {types.shape}")

    anchors_normalized = normalize_anchors(anchors, normalization_info)

    macro_feature = np.load(args.macro_path, allow_pickle=True)
    if macro_feature.ndim != 3 or macro_feature.shape[2] != 2:
        raise ValueError(
            f"macro_feature must have shape (n_traj, T, 2), got {macro_feature.shape}"
        )
    if macro_feature.shape[0] < anchors.shape[0]:
        raise ValueError(
            f"macro_feature has {macro_feature.shape[0]} trajectories, but data has {anchors.shape[0]}"
        )
    if macro_feature.shape[1] <= np.max(time_idx):
        raise ValueError("macro_feature has fewer timesteps than requested.")
    macro_feature = macro_feature[: anchors.shape[0], time_idx, :]

    print("Computing Z for type 1...")
    z_type1 = get_deepset_z_over_time(
        model=model,
        anchors=anchors_normalized,
        types=types,
        particle_type=1,
        device=device,
    )
    print("Computing Z for type 2...")
    z_type2 = get_deepset_z_over_time(
        model=model,
        anchors=anchors_normalized,
        types=types,
        particle_type=2,
        device=device,
    )
    output_path = os.path.join(output_root, args.output_name)
    np.savez_compressed(
        output_path,
        macro_feature=macro_feature,
        z_type1=z_type1,
        z_type2=z_type2,
    )
    print(f"Saved macro/Z arrays to {output_path}")
