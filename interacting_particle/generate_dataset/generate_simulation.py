#!/usr/bin/env python3
"""
Batched (multi-trajectory) particle simulation with Gaussian-mixture initialization.

Notes on shapes:
- M = num_traj
- N = particles per trajectory
- steps = number of Euler steps
- record_every = store every k-th step (k=1 stores all)
- num_T = steps // record_every + 1  (includes the initial state at t=0)

The simulation kernel is vectorized over particles; trajectories are processed in chunks
to keep the (chunk, N, N, 2) working arrays bounded.
"""

from __future__ import annotations

import argparse
import os
import multiprocessing as mp
import numpy as np
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

# -----------------------------
# Force law: Eq. (3)
# F(r) = tanh(a(1-r)) + b
# -----------------------------
def force_F_step(r, a, b):
    return np.tanh(a * (1.0 - r)) + b




def force_dX_step_batch(X: np.ndarray, a: float, b: float, eps: float = 1e-6) -> np.ndarray:
    """
    X: (B,N,2) -> dX: (B,N,2)
    """
    B, N, _ = X.shape
    D = X[:, :, None, :] - X[:, None, :, :]         # (B,N,N,2)
    r2 = np.sum(D * D, axis=-1)                     # (B,N,N)
    r = np.sqrt(r2 + eps**2)                        # (B,N,N)

    # mask self-interactions
    mask = np.eye(N, dtype=bool)[None, :, :]        # (1,N,N)
    F = force_F_step(r, a, b)                       # (B,N,N)
    F = np.where(mask, 0.0, F)

    inv_r = np.where(mask, 0.0, 1.0 / r)
    V = (F * inv_r)[:, :, :, None] * D              # (B,N,N,2)
    dX = (1.0 / N) * np.sum(V, axis=2)              # sum over j -> (B,N,2)
    return dX


def rhs_multi_chunked(X: np.ndarray, a: float, b: float, eps: float, chunk_size: int) -> np.ndarray:
    """
    Compute dX for all trajectories by chunking in the trajectory dimension.
    X: (M,N,2)
    Returns: (M,N,2)
    """
    M = X.shape[0]
    out = np.empty_like(X)
    for s in range(0, M, chunk_size):
        e = min(M, s + chunk_size)
        out[s:e] = force_dX_step_batch(X[s:e], a=a, b=b, eps=eps)
    return out


# -----------------------------
# Initialization (2-component GMM, random means per trajectory)
# -----------------------------
def GMM_init(
    rng: np.random.Generator,
    M: int,
    N: int,
    weights=(1 / 2, 1 / 2),
    covs=None,
    dtype=np.float32,
):
    """
    Sample X from a 2-component Gaussian mixture for M trajectories in parallel.
    Means are drawn from [-1, 1]^2 uniformly for each trajectory.
    Returns:
      X0:    (M,N,2) initial positions
      means: (M,2,2) the sampled component means per trajectory
    """
    K = 2  # number of mixture components

    if len(weights) != K:
        raise ValueError(f"weights must have length {K}")
    if len(covs) != K:
        raise ValueError(f"covs must have length {K}")

    weights_arr = np.asarray(weights, dtype=float)
    if not np.isclose(weights_arr.sum(), 1.0):
        weights_arr = weights_arr / weights_arr.sum()

    covs_arr = np.asarray(covs, dtype=float)

    # Output arrays
    X0 = np.empty((M, N, 2), dtype=dtype)
    means = np.empty((M, K, 2), dtype=dtype)

    for m in range(M):
                
        theta = rng.uniform(0.0, 2*np.pi, size=K)
        R = 1.0
        r = R * np.sqrt(rng.uniform(0.0, 1.0, size=K))  # sqrt -> uniform over area
        x = r * np.cos(theta)
        y = r * np.sin(theta)        
        means_m = np.stack([x, y], axis=1)
        
        means[m] = means_m.astype(dtype, copy=False)

        # Sample component assignments for N particles
        comps_m = rng.choice(K, size=N, p=weights_arr)

        Xm = np.empty((N, 2), dtype=dtype)
        for k in range(K):
            idx = np.where(comps_m == k)[0]
            if idx.size == 0:
                continue
            cov_k = covs_arr[k]
            # Draw from 2D Gaussian with mean means_m[k]
            samples_k = rng.multivariate_normal(mean=means_m[k], cov=cov_k, size=idx.size)
            Xm[idx] = samples_k.astype(dtype, copy=False)

        X0[m] = Xm

    return X0, means


# -----------------------------
# Initialization (3-component GMM, random means per trajectory)
# For generalization test
# -----------------------------
def GMM3(
    rng: np.random.Generator,
    M: int,
    N: int,
    weights=(1 / 3, 1 / 3, 1 / 3),
    covs=None,
    dtype=np.float32,
):
    """
    Sample X from a 3-component Gaussian mixture for M trajectories in parallel.
    Means are drawn from the unit disk for each trajectory.
    Returns:
      X0:    (M,N,2) initial positions
      means: (M,3,2) the sampled component means per trajectory
    """
    K = 3  # number of mixture components

    if len(weights) != K:
        raise ValueError(f"weights must have length {K}")
    if len(covs) != K:
        raise ValueError(f"covs must have length {K}")

    weights_arr = np.asarray(weights, dtype=float)
    if not np.isclose(weights_arr.sum(), 1.0):
        weights_arr = weights_arr / weights_arr.sum()

    covs_arr = np.asarray(covs, dtype=float)

    # Output arrays
    X0 = np.empty((M, N, 2), dtype=dtype)
    means = np.empty((M, K, 2), dtype=dtype)

    for m in range(M):
        # Sample 3 component means from the unit disk
        theta = rng.uniform(0.0, 2 * np.pi, size=K)
        R = 1.0
        r = R * np.sqrt(rng.uniform(0.0, 1.0, size=K))  # sqrt -> uniform over area
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        means_m = np.stack([x, y], axis=1)

        means[m] = means_m.astype(dtype, copy=False)

        # Sample component assignments for N particles
        comps_m = rng.choice(K, size=N, p=weights_arr)

        Xm = np.empty((N, 2), dtype=dtype)
        for k in range(K):
            idx = np.where(comps_m == k)[0]
            if idx.size == 0:
                continue
            cov_k = covs_arr[k]
            # Draw from 2D Gaussian with mean means_m[k]
            samples_k = rng.multivariate_normal(mean=means_m[k], cov=cov_k, size=idx.size)
            Xm[idx] = samples_k.astype(dtype, copy=False)

        X0[m] = Xm

    return X0, means


# -----------------------------
# Simulation + trajectory saving
# -----------------------------
def simulate_euler_multi_to_npy(
    out_npy: str | Path = "trajectories.npy",
    M: int = 100,
    N: int = 300,
    a: float = 6.0,
    b: float = 0.5,
    dt: float = 0.02,
    steps: int = 300,
    eps: float = 1e-6,
    record_every: int = 1,
    seed: int = 1,
    chunk_size: int = 8,
    dtype=np.float32,
    # GMM params:
    gmm_components: int = 2,
    gmm_weights=(1 / 2, 1 / 2),    
    gmm_std=None,

):
    """
    Forward Euler simulation for M trajectories in parallel, writing trajectory tensor
    to a .npy file as it runs.

    Writes:
      out_npy: array of shape (M, num_T, N, 2), dtype=dtype

    Returns:
      out_npy_path (Path), means (M,2,2), comps (M,N)
    """
    out_npy = Path(out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Initialize positions from a Gaussian mixture
    if gmm_components not in (2, 3):
        raise ValueError("gmm_components must be 2 or 3")
    gmm_covs = np.eye(2) * gmm_std  # isotropic covariances with variance gmm_std^2
    # repeat for each component
    gmm_covs = tuple([gmm_covs for _ in range(gmm_components)])

    if gmm_components == 2:
        X, means = GMM_init(
            rng=rng,
            M=M,
            N=N,
            weights=gmm_weights,
            covs=gmm_covs,
            dtype=dtype,
        )
    else:
        X, means = GMM3(
            rng=rng,
            M=M,
            N=N,
            weights=gmm_weights,
            covs=gmm_covs,
            dtype=dtype,
        )

    # ---------------------------------------------------------
    # Recording schedule (piecewise in step index s = t+1):
    #   - for steps in [1, 20],  record every step
    #   - for steps in (20, 100], record every 2 steps
    #   - for steps in (100, 200], record every 5 steps
    #   - for steps  > 200,       record every 20 steps
    # Initial snapshot at t=0 is always recorded.
    # ---------------------------------------------------------
    def _should_record(step_num: int) -> bool:
        if step_num <= 20:
            return True
        elif step_num <= 100:
            return (step_num % 2) == 0
        elif step_num <= 250:
            return (step_num % 5) == 0
        else:
            return (step_num % 20) == 0

    # Precompute total number of recorded time points, including t=0.
    num_T = steps + 1  # start with all steps

    traj = np.empty(shape=(M, num_T, N, 2), dtype=dtype)
    traj[:, 0, :, :] = X



    for t in tqdm(range(steps), desc="Simulating trajectories"):
        dX = rhs_multi_chunked(X, a=a, b=b, eps=eps, chunk_size=chunk_size)
        X = X + dt * dX

        step_num = t + 1
        traj[:, step_num, :, :] = X

    # save to .npy
    np.save(out_npy, traj)
    return traj


def _run_simulation_chunk(args):
    """Worker function to simulate a chunk of trajectories and save to disk.

    Args tuple:
      (chunk_idx, M_chunk, N, a, b, dt, steps, eps, seed, dtype,
       gmm_components, gmm_weights, gmm_std, out_dir)
    """
    (
        chunk_idx,
        M_chunk,
        N,
        a,
        b,
        dt,
        steps,
        eps,
        seed,
        dtype,
        gmm_components,
        gmm_weights,
        gmm_std,
        out_dir,
    ) = args

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = out_dir / f"trajectories_chunk_{chunk_idx}.npy"

    simulate_euler_multi_to_npy(
        out_npy=out_npy,
        M=M_chunk,
        N=N,
        a=a,
        b=b,
        dt=dt,
        steps=steps,
        eps=eps,
        # record_every is ignored by the custom recording schedule
        record_every=1,
        seed=seed,
        chunk_size=M_chunk,
        dtype=dtype,
        gmm_components=gmm_components,
        gmm_weights=gmm_weights,
        gmm_std=gmm_std,
    )

    return str(out_npy)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate interacting-particle simulation trajectories."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("train", "in_dst_test", "diff_init_test", "diff_N_test"),
        required=True,
        help="Configuration mode: train, in_dst_test, diff_init_test, or diff_N_test.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode_config = {
        "train": {
            "N": 300,
            "M_total": 10000,
            "gmm_components": 2,
            "base_seed": 1,
            "out_path": Path("data/trajectories.npy"),
        },
        "in_dst_test": {
            "N": 300,
            "M_total": 1000,
            "gmm_components": 2,
            "base_seed": 1234,
            "out_path": Path("data/trajectories_inDistribution.npy"),
        },
        "diff_init_test": {
            "N": 300,
            "M_total": 1000,
            "gmm_components": 3,
            "base_seed": 1234,
            "out_path": Path("data/trajectories_outDistribution_3gmm.npy"),
        },
        "diff_N_test": {
            "N": 400,
            "M_total": 1000,
            "gmm_components": 2,
            "base_seed": 1234,
            "out_path": Path("data/trajectories_outDistribution_400N.npy"),
        },
    }
    config = mode_config[args.mode]

    # - Save all time steps (record_every=1) into trajectories.npy

    N = config["N"]
    M_total = config["M_total"]

    a = 4
    b = 0.1
    gmm_components = config["gmm_components"]
    gmm_weights = (1 / 2, 1 / 2) if gmm_components == 2 else (1 / 3, 1 / 3, 1 / 3)

    gmm_std = 0.01
    dt = 0.02
    steps = 300
    eps = 1e-6

    # Parallel configuration: split into 10 chunks 
    n_workers = 20
    assert M_total % n_workers == 0, "M_total must be divisible by n_workers"
    M_chunk = M_total // n_workers  

    base_seed = config["base_seed"]
    out_path = config["out_path"]
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    args_list = []
    for i in range(n_workers):
        seed_i = base_seed + i  # distinct, reproducible seed per worker
        args_list.append(
            (
                i,
                M_chunk,
                N,
                a,
                b,
                dt,
                steps,
                eps,
                seed_i,
                np.float32,
                gmm_components,
                gmm_weights,
                gmm_std,
                out_dir,
            )
        )

    with mp.Pool(processes=n_workers) as pool:
        chunk_files = pool.map(_run_simulation_chunk, args_list)

    # Load, concatenate, and save final trajectories array
    traj_list = [np.load(cf) for cf in chunk_files]
    traj = np.concatenate(traj_list, axis=0)
    np.save(out_path, traj)

    # Optionally clean up chunk files
    for cf in chunk_files:
        try:
            os.remove(cf)
        except OSError:
            pass

    print(f"traj shape: {traj.shape}")  # (M_total, num_T, N, 2)


if __name__ == "__main__":
    main()
