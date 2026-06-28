import argparse
import math
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from models import OnsagerNet_original
from models import drift_MLP


def load_checkpoint(model_path, device):
    state = torch.load(model_path, map_location=device, weights_only=False)
    normalization_info = None
    if isinstance(state, dict) and "model_state_dict" in state:
        normalization_info = state.get("normalization_info")
        state = state["model_state_dict"]
    return state, normalization_info


def extract_min_max(normalization_info, z_dim=None):
    if not normalization_info:
        return None, None
    data_min = normalization_info.get("z_min")
    data_max = normalization_info.get("z_max")
    if data_min is None or data_max is None:
        data_min = normalization_info.get("min")
        data_max = normalization_info.get("max")
    if data_min is None or data_max is None:
        return None, None
    data_min = np.asarray(data_min).squeeze()
    data_max = np.asarray(data_max).squeeze()
    if z_dim is not None and data_min.shape[0] != z_dim:
        data_min = data_min.reshape(-1)[-z_dim:]
        data_max = data_max.reshape(-1)[-z_dim:]
    return data_min, data_max


def load_macro_and_z(data_path):
    data_npz = np.load(data_path)
    macro = data_npz["macro"]
    Z = data_npz["Z"]
    if macro.ndim == 2:
        macro = macro[:, :, None]
    return macro, Z


def rk4_step(model, z, dt):
    # Classic fixed-step RK4 update.
    dt = torch.as_tensor(dt, device=z.device, dtype=z.dtype)
    k1 = model(z)
    k2 = model(z + 0.5 * dt * k1)
    k3 = model(z + 0.5 * dt * k2)
    k4 = model(z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def evaluate_ode(
    data_path,
    dt_scalar,
    model_path,
    first_K=11,
    output_path=".",
    drift_model="OnsagerNet",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load data: [num_tra, T, D]
    macro, Z = load_macro_and_z(data_path)

    # data = data[..., 0:1]  # only use first dimension for ODE evaluation

    state_dict, normalization_info = load_checkpoint(model_path, device)
    Z_min, Z_max = extract_min_max(normalization_info, z_dim=Z.shape[-1])
    if Z_min is not None and Z_max is not None:
        Z = 2.0 * (Z - Z_min) / (Z_max - Z_min) - 1.0
    else:
        print("No Z normalization info found in checkpoint; using raw Z.")
    data = np.concatenate([macro, Z], axis=-1)

    _, T, n_dim = data.shape

    # Sort validation trajectories by their initial value in the first dimension.
    # This makes plots easier to compare across runs.
    sort_idx = np.argsort(data[:, 0, 0])
    data = data[sort_idx]
    selected_idx = np.linspace(0, data.shape[0]-1, min(first_K, data.shape[0])).astype(int)


    print(f"data shape: {data.shape}")

    data = torch.from_numpy(data).float().to(device)
    

    # build model and load best weights
    if drift_model == "OnsagerNet":
        model = OnsagerNet_original(input_dim=n_dim).to(device)
    elif drift_model == "MLP":
        model = drift_MLP(input_dim=n_dim).to(device)
    else:
        raise ValueError(f"Unknown drift_model: {drift_model}")
    
    
    model.load_state_dict(state_dict)
    model.eval()

    # ----- vectorized trajectory test for first K experiments -----
    with torch.no_grad():
        n_exp = min(first_K, data.shape[0])
        if n_exp == 0:
            print("No validation trajectories available for plotting.")
        else:
            # select first n_exp trajectories (normalized): [n_exp, T, D]
            # data_sel = data[:n_exp]  # [n_exp, T, D]
            data_sel = data[selected_idx]  # [n_exp, T, D]

            steps = data_sel.shape[1]
            
            
            # initial state for all experiments at t=0: [n_exp, D]
            z_t = data_sel[:, 0, :]  # [n_exp, D]

            preds = [z_t]
            for _ in range(steps - 1):
                z_t = rk4_step(model, z_t, dt_scalar)
                preds.append(z_t)

            pred_tra = torch.stack(preds, dim=1)  # [n_exp, T, D]

            
            # Save comparison plot for the first dimension only.
            t = np.arange(steps) * dt_scalar
            fig, ax = plt.subplots(figsize=(8, 5))
            true_all = data_sel[:, :, 0].detach().cpu().numpy()   # [n_exp, T]
            pred_all = pred_tra[:, :, 0].detach().cpu().numpy()   # [n_exp, T]
            for exp_idx in range(n_exp):
                (true_line,) = ax.plot(t, true_all[exp_idx], label=f"true exp{exp_idx}")
                color = true_line.get_color()
                ax.plot(t, pred_all[exp_idx], linestyle="--", color=color, label=f"pred exp{exp_idx}")
            ax.set_xlabel("time")
            ax.set_ylabel("z[0]")

            fig.tight_layout()
            out_file = os.path.join(output_path, "TEST-true_vs_pred.png")
            fig.savefig(out_file, dpi=300)
            plt.close(fig)
            print(f"Saved comparison plot to {out_file}")


def compute_rollout_loss(
    data_path,
    dt_scalar,
    model_path,
    drift_model="OnsagerNet",
    device=None,
    eps=1e-12,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load data: [num_tra, T, D]
    macro, Z = load_macro_and_z(data_path)
    state_dict, normalization_info = load_checkpoint(model_path, device)
    Z_min, Z_max = extract_min_max(normalization_info, z_dim=Z.shape[-1])
    
    Z_min = np.asarray(Z_min).squeeze()
    Z_max = np.asarray(Z_max).squeeze()
    if Z_min.shape[0] != Z.shape[-1]:
        Z_min = Z_min.reshape(-1)[-Z.shape[-1]:]
        Z_max = Z_max.reshape(-1)[-Z.shape[-1]:]
    Z = 2.0 * (Z - Z_min) / (Z_max - Z_min) - 1.0
    data = np.concatenate([macro, Z], axis=-1)
    _, T, n_dim = data.shape
    if data.shape[0] == 0:
        print("No validation trajectories available for rollout loss.")
        return None

    data_t = torch.from_numpy(data).float().to(device)

    # build model and load best weights
    if drift_model == "OnsagerNet":
        model = OnsagerNet_original(input_dim=n_dim).to(device)
    elif drift_model == "MLP":
        model = drift_MLP(input_dim=n_dim).to(device)
    else:
        raise ValueError(f"Unknown drift_model: {drift_model}")

    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        z_t = data_t[:, 0, :]  # [n_val, D]
        data_macro_0 = data_t[:, :, 0]
        print(data_macro_0[0, 0:10])

        # Only use the first dimension for the rollout loss.
        pred_macro_0 = z_t[:, 0]
        diff = pred_macro_0 - data_macro_0[:, 0]
        numerator = diff ** 2

        for step in range(1, T):
            z_t = rk4_step(model, z_t, dt_scalar)
            pred_macro_0 = z_t[:, 0]
            diff = pred_macro_0 - data_macro_0[:, step]
            numerator += diff ** 2

        denominator = (data_macro_0 ** 2).sum(dim=1)
        rel_error = numerator / (denominator + eps)
        rollout_loss = rel_error.mean().item()

    print(f"Rollout mean relative error (dim 0): {rollout_loss:.6e}")
    return rollout_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_folder",
        type=str,
        default="../trained_nflow_gmm2/exp1/",
        help="Path to .npy file of shape [num_tra, T, D].",
    )
    
    parser.add_argument(
        "--dt",
        type=float,
        default=0.002,
        help="Time step between successive slices in the trajectories.",
    )    

    parser.add_argument("--first_K", type=int, default=5, help="Number of first trajectories to test and plot.")

    parser.add_argument("--drift_model", type=str, default="MLP", choices=["MLP", "OnsagerNet"])
    parser.add_argument(
        "--rollout_loss",
        action="store_true",
        help="Compute rollout mean relative error on validation trajectories.",
    )
    parser.set_defaults(rollout_loss=True)

    parser.add_argument("--device", type=int, default=0, help="CUDA device index to use.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("in_dst_test", "diff_init_test", "diff_N_test"),
        default="in_dst_test",
        help="Configuration mode: in_dst_test, diff_init_test, or diff_N_test.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    mode_to_input_file = {
        "train": "macro_and_Z_over_time.npz",
        "in_dst_test": "test_inDistribution.npz",
        "diff_init_test": "test_outDistribution_3gmm.npz",
        "diff_N_test": "test_outDistribution_400N.npz",
    }
    input_file = mode_to_input_file[args.mode]

    
    data_path = os.path.join(args.base_folder, "macro_input", input_file)
    model_path = os.path.join(args.base_folder, "learned_dynamics", args.drift_model, "best_drift_mlp.pth")
    output_path = os.path.join(args.base_folder, "learned_dynamics", args.drift_model)
    evaluate_ode(
        data_path=data_path,
        dt_scalar=args.dt,
        model_path=model_path,
        first_K=args.first_K,
        output_path=output_path,
        drift_model=args.drift_model,
        device=device,
    )

    if args.rollout_loss:
        compute_rollout_loss(
            data_path=data_path,
            dt_scalar=args.dt,
            model_path=model_path,
            drift_model=args.drift_model,
            device=device,
        )


if __name__ == "__main__":
    main()
