import argparse
import os
import math
import math

from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def cmpt_energy(data_points: np.ndarray):
    # input: data_points: np.ndarray, shape (n_traj, T, N, D)
    # return: macro_feat_E_T: np.ndarray, shape (n_traj, T)
    assert len(data_points.shape) == 4  # (n_traj, T, N, D)
    a = 4.0
    b = 0.1

    def _logcosh(x: torch.Tensor) -> torch.Tensor:
        ax = torch.abs(x)
        return ax + torch.log1p(torch.exp(-2.0 * ax)) - math.log(2.0)

    X = torch.as_tensor(data_points)
    n_traj, T, N, _ = X.shape

    iu = torch.triu_indices(N, N, offset=1, device=X.device)
    chunk_size = 20  # process 20 trajectories at a time
    energy_chunks = []
    for start in tqdm(range(0, n_traj, chunk_size), desc="Computing energy in chunks"):
        end = min(start + chunk_size, n_traj)
        X_chunk = X[start:end]  # (C, T, N, D)
        diff = X_chunk[..., :, None, :] - X_chunk[..., None, :, :]
        r = torch.linalg.norm(diff, dim=-1)
        rij = r[..., iu[0], iu[1]]

        x = a * (1.0 - rij)
        P = (1.0 / a) * _logcosh(x) + b * (1.0 - rij)
        energy = torch.sum(P, dim=-1)  # (C, T)
        energy_chunks.append(energy)

    energy_all = torch.cat(energy_chunks, dim=0)  # (n_traj, T)
    return energy_all.cpu().numpy()
        
    



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate macro-observation features from trajectory data."
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
            "in_npy": "trajectories.npy",
            "out_npy": "macro_feature.npy",
        },
        "in_dst_test": {
            "in_npy": "trajectories_inDistribution.npy",
            "out_npy": "macro_feature_inDistribution.npy",
        },
        "diff_init_test": {
            "in_npy": "trajectories_outDistribution_3gmm.npy",
            "out_npy": "macro_feature_outDistribution_3gmm.npy",
        },
        "diff_N_test": {
            "in_npy": "trajectories_outDistribution_400N.npy",
            "out_npy": "macro_feature_outDistribution_400N.npy",
        },
    }
    config = mode_config[args.mode]

    in_npy = config["in_npy"]
    out_npy = config["out_npy"]

    data_dir = "./data"

    # set input file; if not trajectories.npy, reuse saved normalization stats
    input_file = os.path.join(data_dir, in_npy)

    ## load data
    input_data = np.load(input_file)
    print(f"input_data shape: {input_data.shape}")  # (E,T,N,D)



    print("\n\n get energy over time...")
    macro_feat_E_T = cmpt_energy(
        data_points=input_data, # [E,T,N,D]
    )
        
    print(f"macro_feat_E_T shape: {macro_feat_E_T.shape}")
    
    

    print("\n\n")
    print(f"Final macro feature shape: {macro_feat_E_T.shape}")
    macro_feat_E_T_1 = macro_feat_E_T[:, :, None]

    # divide by the number of particles
    N_particles = input_data.shape[2]
    # assert N_particles == 300, f"Expect N_particles=300, got {N_particles}"
    macro_feat_E_T_1 = macro_feat_E_T_1 / float(N_particles) / float(N_particles - 1) * 2.0  # normalize by N*(N-1)/2 pairs
    

    # normalize macro_feature to [-1, 1] over all experiments and time steps
    if os.path.basename(input_file) != "trajectories.npy":
        print("## load normalization info ...")
        norm_path = os.path.join(data_dir, "macro_feature_normalization.npz")
        norm_data = np.load(norm_path)
        Z_min = norm_data["Z_min"]
        Z_max = norm_data["Z_max"]
    else:
        Z_min = macro_feat_E_T_1.min(axis=(0, 1), keepdims=True)
        Z_max = macro_feat_E_T_1.max(axis=(0, 1), keepdims=True)
    macro_feat_E_T_1 = 2.0 * (macro_feat_E_T_1 - Z_min) / (Z_max - Z_min) - 1.0
    

    # save to npy
    output_npy_path = os.path.join(data_dir, out_npy)
    np.save(output_npy_path, macro_feat_E_T_1)  


    if os.path.basename(input_file) == "trajectories.npy":
        print("## save normalization info ...")
        normalization_info = {
            "Z_min": Z_min,
            "Z_max": Z_max,
        }
        output_norm_path = os.path.join(data_dir, "macro_feature_normalization.npz")
        np.savez(output_norm_path, **normalization_info)
        print(f"Saved macro feature normalization info to: {output_norm_path}")


if __name__ == "__main__":
    main()
