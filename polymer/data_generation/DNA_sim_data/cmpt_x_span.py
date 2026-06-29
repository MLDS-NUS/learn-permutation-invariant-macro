#!/usr/bin/env python3
"""Compute x-span lengths from image trajectories and save in the same format as make_data.py."""
import argparse
import pickle
from typing import Iterable, List

import numpy as np
from tqdm import tqdm


def Image2Length(frames: np.ndarray) -> np.ndarray:
    """Compute x-span per frame from image data.

    Returns a (T, 1) array to match valid_image_length_data.pkl format.
    """
    frames_arr = np.asarray(frames)
    if frames_arr.ndim == 2:
        frames_arr = frames_arr[None, ...]
    elif frames_arr.ndim == 4 and frames_arr.shape[-1] == 1:
        frames_arr = frames_arr[..., 0]
    elif frames_arr.ndim != 3:
        raise ValueError(f"Expected frames with 2-4 dims, got shape {frames_arr.shape}.")

    lengths = np.zeros((frames_arr.shape[0], 1), dtype=np.float32)
    for i, frame in enumerate(frames_arr):
        mask = frame > 0
        if not np.any(mask):
            lengths[i, 0] = 0.0
            continue
        cols = np.where(mask)[1].astype(np.int64)
        if cols.size == 0:
            lengths[i, 0] = 0.0
            continue
        lengths[i, 0] = float(cols.max() - cols.min())
    return lengths


def compute_all_lengths(image_data: Iterable[np.ndarray]) -> List[np.ndarray]:
    return [Image2Length(traj_frames) for traj_frames in tqdm(image_data, desc="Computing x-span")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_pkl",
        type=str,
        default="Data/ValidData/valid_image_data.pkl",
        help="Path to *_image_data.pkl file.",
    )
    p.add_argument(
        "--out_pkl",
        type=str,
        default="Data/ValidData/valid_image_length_data.pkl",
        help="Output pickle path for image length data.",
    )
    args = p.parse_args()

    with open(args.input_pkl, "rb") as f:
        image_data = pickle.load(f)

    image_length_data = compute_all_lengths(image_data)

    with open(args.out_pkl, "wb") as f:
        pickle.dump(image_length_data, f)

    print(f"Saved image length data to {args.out_pkl}")


if __name__ == "__main__":
    main()
