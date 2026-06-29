#!/usr/bin/env python3
"""Generate set-encoder Z embeddings and x-span features from pixel-set caches."""
import os
import argparse
from typing import Iterable, Tuple

import numpy as np
import torch

from models_image import SetEncoderND
from train_nflow_on_pixels import NFlowsConditionalARQSPixels
from tqdm import tqdm

def compute_feature_minmax(coords_px: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coords_px = np.asarray(coords_px, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    x = coords_px[:, 0]
    y = coords_px[:, 1]
    feature_min = np.array([x.min(), y.min(), weights.min()], dtype=np.float32)
    feature_max = np.array([x.max(), y.max(), weights.max()], dtype=np.float32)
    return feature_min, feature_max


def compute_x_span(coords_px: np.ndarray, weights: np.ndarray) -> float:
    if coords_px.size == 0:
        return 0.0
    valid = weights > 0
    if not np.any(valid):
        return 0.0
    cols = np.floor(coords_px[valid, 0]).astype(np.int64)
    if cols.size == 0:
        return 0.0
    return float(cols.max() - cols.min())


def load_npz_arrays(npz_path: str):
    npz = np.load(npz_path)
    coords_all = np.array(npz["coords"], copy=False)
    weights_all = np.array(npz["weights"], copy=False)
    frame_ptr = np.array(npz["frame_ptr"], copy=False)
    traj_uid = np.array(npz["traj_uid"], copy=False)
    t_arr = np.array(npz["t"], copy=False)
    H = int(npz["H"])
    W = int(npz["W"])
    return coords_all, weights_all, frame_ptr, traj_uid, t_arr, H, W


def build_model_from_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_hparams = ckpt.get("model_hparams", {})
    required = {"dim", "z_dim", "hidden_dim", "n_layers", "rqs_K", "rqs_B"}
    if not required.issubset(model_hparams):
        missing = sorted(required - set(model_hparams))
        raise SystemExit(
            f"Checkpoint missing model_hparams keys: {missing}. Re-train to store hyperparameters."
        )

    dim = int(model_hparams.get("dim", model_hparams.get("feature_dim", 3)))
    z_dim = int(model_hparams["z_dim"])
    hidden_dim = int(model_hparams["hidden_dim"])
    n_layers = int(model_hparams["n_layers"])
    rqs_K = int(model_hparams["rqs_K"])
    rqs_B = float(model_hparams["rqs_B"])
    pool = str(model_hparams.get("deepset_pool", "mean"))
    encoder_type = str(model_hparams.get("encoder_type", "deepset"))

    num_heads = int(model_hparams.get("st_num_heads", model_hparams.get("set_transformer_num_heads", 4)))
    num_layers = int(model_hparams.get("st_num_layers", model_hparams.get("set_transformer_num_layers", 2)))
    num_seeds = int(model_hparams.get("st_num_seeds", model_hparams.get("set_transformer_num_seeds", 1)))
    ff_hidden_dim = model_hparams.get("st_ff_hidden_dim", model_hparams.get("set_transformer_ff_hidden_dim"))
    if ff_hidden_dim is not None:
        ff_hidden_dim = int(ff_hidden_dim)
        if ff_hidden_dim <= 0:
            ff_hidden_dim = None
    dropout = float(model_hparams.get("st_dropout", model_hparams.get("set_transformer_dropout", 0.0)))

    set_encoder = SetEncoderND(
        in_dim=dim,
        hidden_dim=hidden_dim,
        z_dim=z_dim,
        pool=pool,
        encoder_type=encoder_type,
        num_heads=num_heads,
        num_layers=num_layers,
        num_seeds=num_seeds,
        ff_hidden_dim=ff_hidden_dim,
        dropout=dropout,
    ).to(device)
    model = NFlowsConditionalARQSPixels(
        dim=dim,
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        K=rqs_K,
        B=rqs_B,
        set_encoder=set_encoder,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    normalization_info = ckpt.get("normalization_info", {})
    feature_min = normalization_info.get("feature_min")
    feature_max = normalization_info.get("feature_max")
    return model, ckpt, feature_min, feature_max


def pad_batch(coords_list, weights_list):
    sizes = [coords.shape[0] for coords in coords_list]
    Mmax = max(sizes) if sizes else 0
    E = len(coords_list)
    coords_px = torch.zeros((E, Mmax, 2), dtype=torch.float32)
    weights = torch.zeros((E, Mmax), dtype=torch.float32)
    mask = torch.zeros((E, Mmax), dtype=torch.bool)
    for i, (coords, w) in enumerate(zip(coords_list, weights_list)):
        m = coords.shape[0]
        coords_px[i, :m] = torch.from_numpy(coords).float()
        weights[i, :m] = torch.from_numpy(w).float()
        mask[i, :m] = True
    return coords_px, weights, mask


def normalize_to_unit(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    x_min = np.asarray(x_min, dtype=np.float32)
    x_max = np.asarray(x_max, dtype=np.float32)
    denom = np.maximum(x_max - x_min, 1e-6)
    return (x - x_min) / denom * 2.0 - 1.0


def resolve_frames(
    traj_uid: np.ndarray,
    t_arr: np.ndarray,
    *,
    min_t: int | None,
    max_t: int | None,
) -> np.ndarray:
    valid = np.ones_like(t_arr, dtype=bool)
    if min_t is not None:
        valid &= t_arr >= min_t
    if max_t is not None:
        valid &= t_arr <= max_t
    frames = np.flatnonzero(valid)
    if frames.size == 0:
        raise ValueError("No frames available after applying min_t/max_t filtering.")
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_split", type=str, default="valid", choices=["train", "valid", "test_fast", "test_medium", "test_slow"])
    p.add_argument(
        "--ckpt",
        type=str,
        default="trained_nflow_images/Z2/",
    )
    p.add_argument("--min_t", type=int, default=0)
    p.add_argument("--max_t", type=int, default=1000)
    p.add_argument("--batch_frames", type=int, default=256)
    p.add_argument("--device", type=int, default=0)
    
    
    args = p.parse_args()

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    )

    # input paths
    data_npz = f"data_generation/dataset/{args.data_split}_pixel_sets.npz"
    coords_all, weights_all, frame_ptr, traj_uid, t_arr, H, W = load_npz_arrays(data_npz)
    frames = resolve_frames(traj_uid, t_arr, min_t=args.min_t, max_t=args.max_t)

    

    model, ckpt, feature_min, feature_max = build_model_from_checkpoint(os.path.join(args.ckpt, "best_model.pth"), device)



    # output paths
    model_hparams = ckpt.get("model_hparams", {})
    z_dim = int(model_hparams["z_dim"])
    output_dir = os.path.join(args.ckpt, f"macro_data_Z{z_dim}")
    os.makedirs(output_dir, exist_ok=True)
    out_npz_path = os.path.join(output_dir, f"{args.data_split}_Z_data.npz")

    if feature_min is None or feature_max is None:
        raise ValueError("Checkpoint missing normalization_info; cannot proceed.")
        feature_min, feature_max = compute_feature_minmax(coords_all, weights_all)
    feature_min = np.asarray(feature_min, dtype=np.float32)
    feature_max = np.asarray(feature_max, dtype=np.float32)
    feature_min_t = torch.as_tensor(feature_min, dtype=torch.float32, device=device)
    feature_max_t = torch.as_tensor(feature_max, dtype=torch.float32, device=device)
    feature_range_t = (feature_max_t - feature_min_t).clamp_min(1e-6)

    z_dim = int(model_hparams["z_dim"])
    num_frames = int(frames.shape[0])
    z_frames = np.zeros((num_frames, z_dim), dtype=np.float32)
    x_span_frames = np.zeros((num_frames,), dtype=np.float32)

    for start in tqdm(range(0, num_frames, args.batch_frames), desc="Computing Z embeddings"):
        batch_idx = frames[start:start + args.batch_frames]
        coords_list = []
        weights_list = []
        for frame_idx in batch_idx:
            a = int(frame_ptr[frame_idx])
            b = int(frame_ptr[frame_idx + 1])
            coords = coords_all[a:b].astype(np.float32, copy=False)
            weights = weights_all[a:b].astype(np.float32, copy=False)
            coords_list.append(coords)
            weights_list.append(weights)
            x_span_frames[start + len(coords_list) - 1] = compute_x_span(coords, weights)

        coords_px, weights, mask = pad_batch(coords_list, weights_list)
        coords_px = coords_px.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        features = torch.cat([coords_px, weights.unsqueeze(-1)], dim=-1)
        features = (features - feature_min_t) / feature_range_t * 2.0 - 1.0

        with torch.no_grad():
            z = model.set_encoder(features, mask=mask).cpu().numpy()
        z_frames[start:start + z.shape[0]] = z.astype(np.float32, copy=False)

    t_selected = t_arr[frames]
    traj_selected = traj_uid[frames]
    t_values = np.unique(t_selected)
    traj_values = np.unique(traj_selected)

    t_to_idx = {int(t): i for i, t in enumerate(t_values.tolist())}
    traj_to_idx = {int(u): i for i, u in enumerate(traj_values.tolist())}

    n_traj = int(traj_values.shape[0])
    T = int(t_values.shape[0])
    z_raw = np.full((n_traj, T, z_dim), np.nan, dtype=np.float32)
    x_span_raw = np.full((n_traj, T), np.nan, dtype=np.float32)

    for i in range(num_frames):
        t_idx = t_to_idx[int(t_selected[i])]
        u_idx = traj_to_idx[int(traj_selected[i])]
        z_raw[u_idx, t_idx] = z_frames[i]
        x_span_raw[u_idx, t_idx] = x_span_frames[i]

    if np.isnan(z_raw).any() or np.isnan(x_span_raw).any():
        raise ValueError(
            "Detected missing frames when building [n_traj, T, z_dim]. "
            "Check that all trajectories share the same time steps after filtering."
        )

    if args.data_split == "train":
        z_min = z_raw.min(axis=(0, 1))
        z_max = z_raw.max(axis=(0, 1))
        x_span_min = float(x_span_raw.min())
        x_span_max = float(x_span_raw.max())
        print("Training data Z min/max:", z_min, z_max)
        print("Training data x-span min/max:", x_span_min, x_span_max)
    else:
        # load from training data stats
        train_npz_path = os.path.join(output_dir, "train_Z_data.npz")
        assert os.path.isfile(train_npz_path), f"Training Z data not found: {train_npz_path}"            
        train_npz = np.load(train_npz_path)
        z_min = train_npz["z_min"]
        z_max = train_npz["z_max"]
        x_span_min = float(train_npz["x_span_min"])
        x_span_max = float(train_npz["x_span_max"])

    z_norm = normalize_to_unit(z_raw, z_min, z_max)
    x_span_norm = normalize_to_unit(x_span_raw, x_span_min, x_span_max)

    np.savez(
        out_npz_path,
        z_raw=z_raw,
        z_norm=z_norm,
        x_span_raw=x_span_raw,
        x_span_norm=x_span_norm,
        z_min=z_min,
        z_max=z_max,
        x_span_min=np.float32(x_span_min),
        x_span_max=np.float32(x_span_max),
        t_values=t_values.astype(np.int32, copy=False),
        traj_uid=traj_values.astype(np.int32, copy=False),
        min_t=np.int32(args.min_t) if args.min_t is not None else -1,
        max_t=np.int32(args.max_t) if args.max_t is not None else -1,
        feature_min=feature_min,
        feature_max=feature_max,
        H=np.int32(H),
        W=np.int32(W),
    )
    print(f"Saved {out_npz_path}")


if __name__ == "__main__":
    main()
