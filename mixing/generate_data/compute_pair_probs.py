#!/usr/bin/env python3
"""
Compute P_AB and P_BA from saved trajectories in an NPZ file.

P_AB = (total number of B-neighbor "ends" around A atoms) /
       (total neighbor "ends" around A atoms)
P_BA is defined analogously.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

MODE_PATHS = {
    "train": (
        Path("dataset") / "trajectories_large.npz",
        Path("dataset") / "macro_feature_large.npy",
    ),
    "in_dst_test": (
        Path("dataset") / "trajectories_inDistribution_test.npz",
        Path("dataset") / "macro_feature_inDistribution_test.npy",
    ),
    "test_left": (
        Path("dataset") / "trajectories_test_left.npz",
        Path("dataset") / "macro_feature_test_left.npy",
    ),
    "test_mid": (
        Path("dataset") / "trajectories_test_mid.npz",
        Path("dataset") / "macro_feature_test_mid.npy",
    ),
    "test_right": (
        Path("dataset") / "trajectories_test_right.npz",
        Path("dataset") / "macro_feature_test_right.npy",
    ),
    "diff_dst_test": (
        Path("dataset") / "trajectories_diffInitPos_test.npz",
        Path("dataset") / "macro_feature_diffInitPos_test.npy",
    ),
    "diff_N_test": (
        Path("dataset") / "trajectories_diffN_test.npz",
        Path("dataset") / "macro_feature_diffN_test.npy",
    ),
}

_WORKER_TRAJECTORIES: np.ndarray | None = None
_WORKER_TYPES: np.ndarray | None = None
_WORKER_RC: float | None = None


def _init_pair_prob_worker(
    trajectories: np.ndarray,
    types: np.ndarray,
    rc: float,
) -> None:
    global _WORKER_TRAJECTORIES, _WORKER_TYPES, _WORKER_RC
    _WORKER_TRAJECTORIES = trajectories
    _WORKER_TYPES = types
    _WORKER_RC = rc


def _pair_probs_for_trajectory(m: int) -> np.ndarray:
    if _WORKER_TRAJECTORIES is None or _WORKER_TYPES is None or _WORKER_RC is None:
        raise RuntimeError("Worker globals not initialized.")
    traj = _WORKER_TRAJECTORIES[m]
    ttypes = _WORKER_TYPES[m]
    n_steps = traj.shape[0]
    out = np.empty((n_steps, 2), dtype=np.float64)
    for t in range(n_steps):
        p_ab, p_ba = _pair_probs_for_frame(traj[t], ttypes, _WORKER_RC)
        out[t, 0] = p_ab
        out[t, 1] = p_ba
    return out


def _pair_probs_for_frame(xy: np.ndarray, types: np.ndarray, rc: float) -> tuple[float, float]:
    tree = cKDTree(xy)
    neighbors = tree.query_ball_point(xy, r=rc)

    a_total = 0
    a_unlike = 0
    b_total = 0
    b_unlike = 0

    for i, neigh in enumerate(neighbors):
        if i in neigh:
            neigh = [j for j in neigh if j != i]
        if not neigh:
            continue

        ti = types[i]
        neigh_types = types[neigh]
        n_all = len(neigh)

        if ti == 1:
            a_total += n_all
            a_unlike += int(np.sum(neigh_types == 2))
        elif ti == 2:
            b_total += n_all
            b_unlike += int(np.sum(neigh_types == 1))
        else:
            continue

    p_ab = (a_unlike / a_total) if a_total > 0 else float("nan")
    p_ba = (b_unlike / b_total) if b_total > 0 else float("nan")
    return p_ab, p_ba


def compute_pair_probs(
    trajectories: np.ndarray,
    types: np.ndarray,
    rc: float,
    workers: int | None = None,
) -> np.ndarray:
    """
    trajectories: (M, T, N, 2)
    types:        (M, N)
    Returns: (M, T, 2) with [P_AB, P_BA]
    """
    n_traj, n_steps, _, _ = trajectories.shape
    out = np.empty((n_traj, n_steps, 2), dtype=np.float64)

    if workers is None or workers <= 1:
        for m in tqdm(range(n_traj), desc="Computing pair probabilities"):
            ttypes = types[m]
            for t in range(n_steps):
                p_ab, p_ba = _pair_probs_for_frame(trajectories[m, t], ttypes, rc)
                out[m, t, 0] = p_ab
                out[m, t, 1] = p_ba
    else:
        with mp.Pool(
            processes=workers,
            initializer=_init_pair_prob_worker,
            initargs=(trajectories, types, rc),
        ) as pool:
            # Pool.imap preserves input order, keeping trajectory order consistent.
            results = tqdm(
                pool.imap(_pair_probs_for_trajectory, range(n_traj)),
                total=n_traj,
                desc="Computing pair probabilities",
            )
            for m, traj_probs in enumerate(results):
                out[m] = traj_probs

    return out




def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Compute P_AB and P_BA for each frame.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=tuple(MODE_PATHS),
        default="train",
        help="Dataset preset to use for the default input/output paths.",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Input NPZ file. Overrides the path implied by --mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPY file. Overrides the path implied by --mode.",
    )
    parser.add_argument("--rc", type=float, default=2.5, help="Neighbor cutoff radius.")
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of worker processes (0/1 disables multiprocessing).",
    )
    args = parser.parse_args()
    mode_npz, mode_output = MODE_PATHS[args.mode]
    args.npz = args.npz or (script_dir / mode_npz)
    args.output = args.output or (script_dir / mode_output)

    data = np.load(args.npz, allow_pickle=True)
    trajectories = data["trajectories"]
    types = data["types"]
    print(f"trajectories shape: {trajectories.shape}, types shape: {types.shape}")

    pair_probs = compute_pair_probs(trajectories, types, args.rc, args.workers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, pair_probs)

    print(
        f"Saved pair_probs with shape {pair_probs.shape} to {args.output} "

    )


if __name__ == "__main__":
    main()
