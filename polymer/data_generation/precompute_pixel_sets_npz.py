#!/usr/bin/env python3
"""Precompute pixel sets (coords, weights) for every frame in local .npy
image trajectories and save to a single .npz for fast training.

Output arrays in the NPZ:
  coords    : (P, 2) float32   concatenated pixel coordinates per point
  weights   : (P,)   float32   concatenated weights per point
  frame_ptr : (F+1,) int64     prefix sum offsets into coords/weights for each frame
  traj_uid  : (F,)   int32     unique trajectory id (across all trajectories)
  t         : (F,)   int32     timestep within trajectory
  H, W      : int32 scalars
  input_dir : (1,)   U         source directory
  traj_files: (N,)   U         sorted .npy filenames

This matches your current extraction logic in train_image.py.
"""
import os
import argparse
import re
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm
import numpy as np
import pickle as pkl
def extract_pixel_set(
    frame: np.ndarray,
    *,
    white_value: int = 255,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
        
    assert frame.ndim == 2
    frame = 255 - frame  # invert so foreground is bright !!

    A = frame.astype(np.float32)
    mask = A < white_value
    ys, xs = np.where(mask)  # row, col
    if ys.size == 0:
        return (np.array([[0.0, 0.0]], dtype=np.float32),
                np.array([1.0], dtype=np.float32))

    vals = A[ys, xs]
    w = (float(white_value) - vals).astype(np.float32)
    coords = np.stack([xs.astype(np.float32) + 0.5,
                       ys.astype(np.float32) + 0.5], axis=1)

    return coords, w

def _natural_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.name)
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def _iter_frames(frames: np.ndarray):
    if frames.ndim == 2:
        yield frames
        return
    if frames.ndim in (3, 4):
        for t in range(frames.shape[0]):
            yield frames[t]
        return
    raise ValueError(f"Expected frames with 2-4 dims, got shape {frames.shape}.")


def _frame_hw(frame: np.ndarray) -> Tuple[int, int]:
    if frame.ndim == 2:
        return int(frame.shape[0]), int(frame.shape[1])
    if frame.ndim == 3:
        return int(frame.shape[0]), int(frame.shape[1])
    raise ValueError(f"Expected frame with 2-3 dims, got shape {frame.shape}.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_pkl', type=str, default="./DNA_sim_data/Data/TrainData/train_image_data.pkl")
    p.add_argument('--out_npz', type=str, default="dataset/train_pixel_sets.npz")
    p.add_argument('--white_value', type=int, default=255)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--use_compressed', action='store_true',
                   help='Use np.savez_compressed (smaller but slower to load).')
    args = p.parse_args()

    

    os.makedirs(os.path.dirname(args.out_npz) or '.', exist_ok=True)

    coords_chunks: List[np.ndarray] = []
    weights_chunks: List[np.ndarray] = []
    frame_ptr: List[int] = [0]
    traj_uid_list: List[int] = []
    t_list: List[int] = []

    H = 100
    W = 500

    with open(args.input_pkl, 'rb') as f:
        train_image_data = pkl.load(f)
    print(type(train_image_data))

    for traj_uid, frames in tqdm(enumerate(train_image_data), desc='Processing trajectories'):

        if H is None:
            H, W = _frame_hw(next(_iter_frames(frames)))
        else:
            f0 = next(_iter_frames(frames))
            H0, W0 = _frame_hw(f0)
            if (H0, W0) != (H, W):
                raise ValueError(
                    f"Frame size mismatch in trajectory {traj_uid}: "
                    f"expected {(H, W)}, got {(H0, W0)}"
                )

        for t, frame in enumerate(_iter_frames(frames)):
            coords, w = extract_pixel_set(
                frame,
                white_value=args.white_value,
                seed=(args.seed + traj_uid * 100000 + t),
            )
            coords_chunks.append(coords)
            weights_chunks.append(w)
            frame_ptr.append(frame_ptr[-1] + coords.shape[0])
            traj_uid_list.append(traj_uid)
            t_list.append(t)

    coords_all = np.concatenate(coords_chunks, axis=0).astype(np.float32, copy=False)
    weights_all = np.concatenate(weights_chunks, axis=0).astype(np.float32, copy=False)
    frame_ptr_arr = np.asarray(frame_ptr, dtype=np.int64)
    traj_uid_arr = np.asarray(traj_uid_list, dtype=np.int32)
    t_arr = np.asarray(t_list, dtype=np.int32)

    save_fn = np.savez_compressed if args.use_compressed else np.savez
    save_fn(
        args.out_npz,
        coords=coords_all,
        weights=weights_all,
        frame_ptr=frame_ptr_arr,
        traj_uid=traj_uid_arr,
        t=t_arr,
        H=np.int32(H),
        W=np.int32(W),
        white_value=np.int32(args.white_value),
    )

    F = len(traj_uid_arr)
    P = coords_all.shape[0]
    print(f'Saved: {args.out_npz}')
    print(f'Frames: {F:,}  Total points: {P:,}  Avg points/frame: {P / max(F,1):.1f}')
    print(f'Image size: H={H}, W={W}')

if __name__ == '__main__':
    main()
