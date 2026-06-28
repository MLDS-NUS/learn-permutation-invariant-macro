import os
import math
import argparse
import math

from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import (
    SetEncoderND,
)
from train_nflow_arqs import NFlowsConditionalARQS  # AR RQS wrapper from training script




# ---------------------------------------------------------------------------
# Rebuild model + scale from checkpoint (RQS or ARQS)
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, dict, torch.Tensor]:
    """
    Reconstruct the trained model (RQS or ARQS) from the checkpoint and
    also reconstruct the scale normalization used during training.

    Returns:
      model:  nn.Module in eval mode
      ckpt:   checkpoint dict
      scale:  (D,) tensor used for scaling x and anchors
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    dim       = ckpt["dim"]
    model_type = ckpt.get("model_type", "ARQS").upper()
    z_dim     = ckpt.get("z_dim", ckpt.get("args", {}).get("z_dim", 4))
    hidden    = ckpt.get("hidden_dim", ckpt.get("args", {}).get("hidden_dim", 256))
    n_layers  = ckpt.get("n_layers", ckpt.get("args", {}).get("n_layers", 8))
    rqs_K     = ckpt.get("rqs_K", ckpt.get("args", {}).get("rqs_K", 16))
    rqs_B     = ckpt.get("rqs_B", ckpt.get("args", {}).get("rqs_B", 5.0))

    print(f"Checkpoint loaded from: {ckpt_path}")
    print(f"dim={dim}, z_dim={z_dim}, hidden_dim={hidden}, n_layers={n_layers}, "
          f"rqs_K={rqs_K}, rqs_B={rqs_B}, model_type={model_type}")

    # Shared DeepSet encoder
    set_encoder = SetEncoderND(
        in_dim=dim,
        hidden_dim=hidden,
        z_dim=z_dim,
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

    # Reconstruct scale normalization used in training:
    # scale = [rqs_B / x_range, rqs_B / y_range]  (for D=2), from CLI args.
    normalization_info = ckpt.get("normalization_info", ckpt.get("args", {}).get("normalization_info", None))
    # print(f"\nNormalization info from checkpoint args: {normalization_info}")
    # ranges = normalization_info.get("ranges")

    # scale = 1 / ranges
    # scale = torch.from_numpy(scale).float().to(device)

    # print(f"Using evaluation scale: {scale.cpu().numpy()}")
    return model, ckpt, normalization_info



   


def get_deepset_z_over_time(
    model, 
    anchors: np.ndarray,  # (E, T, M, D)
    device,
):
        
    anchors_scaled = torch.from_numpy(anchors).float().to(device)  # (E, T, M, D)
    E, T, N, D = anchors_scaled.shape
    # ------------------------------------------------------------------
    # Run DeepSet encoder over time: Z(e,t,:) = set_encoder(anchors_e_t)
    # ------------------------------------------------------------------
    set_encoder = model.set_encoder
    set_encoder.eval()
    model.eval()

    Z_list = []
    with torch.no_grad():
        for e in tqdm(range(E), desc="Iterating experiments for Z over time"):
            anchors_e = anchors_scaled[e].contiguous() # (T,M,D)
            Z_e = set_encoder(anchors_e)  # (T,z_dim)
            Z_list.append(Z_e.cpu().numpy())
    Z = np.stack(Z_list, axis=0)  # (E,T,z_dim)
    return Z






def plot_deepset_z_over_time( Z: np.ndarray, num_experiments_to_plot: int = 3, max_dims_to_plot: int = None, output_dir: str = "evaluation_nflow"):
    # Z: (E, T, D)
    
    E, T, z_dim = Z.shape
        

    # ------------------------------------------------------------------
    # Decide which dimensions to plot
    # ------------------------------------------------------------------
    if max_dims_to_plot is None:
        num_dims = z_dim
    else:
        num_dims = min(z_dim, max_dims_to_plot)

    num_lines = min(num_experiments_to_plot, E)
    t_axis = np.arange(T)

    

    # ------------------------------------------------------------------
    # Subplot layout: n_rows x 4 columns
    # ------------------------------------------------------------------
    n_cols = 4
    n_rows = math.ceil(num_dims / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        sharex=True,
        constrained_layout=True,
    )

    # Make axes a 2D array
    axes = np.array(axes)
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)

    # Plot each dimension in its own subplot
    for d_idx in range(num_dims):
        row = d_idx // n_cols
        col = d_idx % n_cols
        ax = axes[row, col]

        for e_idx in range(num_lines):
            y = Z[e_idx, :, d_idx]  # (T,)
            # no line, only scatter
            # ax.scatter(t_axis, y, label=f"Exp {e_idx}", s=3)
            ax.plot(t_axis, y, label=f"Exp {e_idx}", linewidth=1)

        ax.set_title(f"z[{d_idx}]")
        ax.grid(alpha=0.3)

        # Add legend only once to avoid clutter
        # if d_idx == 0:
        #     ax.legend()

    # Turn off any unused axes (if num_dims is not a multiple of 4)
    total_slots = n_rows * n_cols
    for empty_idx in range(num_dims, total_slots):
        row = empty_idx // n_cols
        col = empty_idx % n_cols
        axes[row, col].axis("off")

    # Label only bottom row with x-axis
    for col in range(n_cols):
        axes[n_rows - 1, col].set_xlabel("Time step")

    out_path = os.path.join(output_dir, f"dynamics_input_{z_dim}.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained conditional NF (RQS/ARQS)")
    parser.add_argument(
        "--model_path",
        type=str,
        default="trained_nflow_gmm2/exp1/",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="generate_dataset/data/",
        help="path to trajectories.npy"
    )
    parser.add_argument(
        "--time_step",
        type=int,
        nargs="+",
        default=[0, 10, 20, 50, -1],
        help="List of time steps to evaluate, e.g. --time_step 0 100 -1",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("train", "in_dst_test", "diff_init_test", "diff_N_test"),
        required=True,
        help="Configuration mode: train, in_dst_test, diff_init_test, or diff_N_test.",
    )
    parser.add_argument("--num_experiments_to_plot", type=int, default=6)
    parser.add_argument("--kl_samples", type=int, default=4000)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    mode_config = {
        "train": {
            "traj_file": "trajectories.npy",
            "macro_obs_file": "macro_feature.npy",
            "out_npz": "macro_and_Z_over_time.npz",
        },
        "in_dst_test": {
            "traj_file": "trajectories_inDistribution.npy",
            "macro_obs_file": "macro_feature_inDistribution.npy",
            "out_npz": "test_inDistribution.npz",
        },
        "diff_init_test": {
            "traj_file": "trajectories_outDistribution_3gmm.npy",
            "macro_obs_file": "macro_feature_outDistribution_3gmm.npy",
            "out_npz": "test_outDistribution_3gmm.npz",
        },
        "diff_N_test": {
            "traj_file": "trajectories_outDistribution_400N.npy",
            "macro_obs_file": "macro_feature_outDistribution_400N.npy",
            "out_npz": "test_outDistribution_400N.npz",
        },
    }
    config = mode_config[args.mode]

    traj_file = config["traj_file"]
    macro_obs_file = config["macro_obs_file"]
    out_npz = config["out_npz"]

    input_data = np.load(os.path.join(args.data_path, traj_file))
    print(f"input_data shape: {input_data.shape}")  # (E,T,N,D)

    ## load model

    # Load model + scale
    device = torch.device(
        f"cuda:{args.device}"
        if (args.device >= 0 and torch.cuda.is_available())
        else "cpu"
    )
    ckpt_path = os.path.join(args.model_path, "best_model_Z8.pth")
    model, ckpt, normalization_info = build_model_from_checkpoint(ckpt_path, device)
    dim = ckpt["dim"]
    print(f"normalization_info: {normalization_info}")
    

    ## normalize input data
    # Apply same scaling as in training
    mean = normalization_info.get("mean").squeeze()
    std = normalization_info.get("std").squeeze()
    anchors_normalized = (input_data - mean) / std  # (E,T,N,D)


    print("\n\n get energy over time...")
    
    macro_feat_E_T = np.load(os.path.join(args.data_path, macro_obs_file)).squeeze()
    print(f"macro_feat_E_T shape: {macro_feat_E_T.shape}")
    
    print("\n\n get Z over time...")
    Z_E_T_D = get_deepset_z_over_time(
        model=model,
        anchors=anchors_normalized,
        device=device,
    )
    print(f"Z_E_T_D shape: {Z_E_T_D.shape}")
    
           
    output_dir = os.path.join(args.model_path, "macro_input")
    os.makedirs(output_dir, exist_ok=True)

    print("\n\n")
    print(f"Final macro feature shape: {macro_feat_E_T.shape}")
    print(f"Final Z shape: {Z_E_T_D.shape}")
    macro_feat_E_T_1 = macro_feat_E_T[:, :, None]
    plot_data = np.concatenate([macro_feat_E_T_1, Z_E_T_D], axis=-1)

    # plot the first 10 experiments' macro-feature and Z over time
    plot_deepset_z_over_time(
        Z=plot_data,
        num_experiments_to_plot=args.num_experiments_to_plot,
        output_dir=output_dir,
    )

    # save macro and Z separately
    output_npz_path = os.path.join(output_dir, out_npz)
    np.savez(
        output_npz_path,
        macro=macro_feat_E_T_1,
        Z=Z_E_T_D,
    )
    # Normalization is now handled during ODE training.


if __name__ == "__main__":
    main()
