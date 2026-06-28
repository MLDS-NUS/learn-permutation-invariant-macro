import os
import argparse
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Import model classes from your training module
from train import (
    SetEncoder,
    BaseHead,
    ConditionalFlow1D,
    sample_from_anchor_mixture_batch,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--anchors_path", type=str, default="data_generation/gaussian_mixture_data_test.pt",
                        help="Path to .pt file containing 'anchors' tensor.")
    parser.add_argument("--exp_ids", type=int, nargs="+", default=[10],
                        help="List of experiment IDs to visualize.")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Std for local Gaussians in the true mixture.")
    parser.add_argument("--z_dim", type=int, default=8,
                        help="Latent dimension of the checkpoint to evaluate.")
    parser.add_argument("--n_grid_points", type=int, default=500,
                        help="Number of x points in the grid for plotting.")
    parser.add_argument("--device", type=str, default="auto",
                        help='Device: "auto", "cpu", or "cuda".')
    
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utility: underlying Gaussian mixture to draw anchors from
# ---------------------------------------------------------------------------
def gaussian_pdf(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Compute 1D Gaussian PDF for possibly vectorized x, mean, std (numpy)."""
    var = std ** 2
    return 1.0 / np.sqrt(2.0 * np.pi * var) * np.exp(-0.5 * (x - mean) ** 2 / var)


def mixture_pdf(x: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Uniform-weight Gaussian mixture PDF evaluated on grid x."""
    # x: (N,), means/stds: (K,)
    # Broadcast to (N, K)
    x_expanded = x[:, None]
    means_expanded = means[None, :]
    stds_expanded = stds[None, :]
    comps = gaussian_pdf(x_expanded, means_expanded, stds_expanded)
    return comps.mean(axis=1)



# ---------------------------------------------------------------------------
# Utility: true mixture density
# ---------------------------------------------------------------------------

def mixture_target_pdf(x_grid, anchors_exp, epsilon):
    """
    True target mixture for one experiment:
        p(x|anchors) = (1/M) sum_j N(x | anchors_j, epsilon^2)
    """
    x = x_grid.unsqueeze(1)                 # (G, 1)
    anchors_flat = anchors_exp.squeeze(-1).unsqueeze(0)  # (1, M)
    diff = x - anchors_flat                 # (G, M)

    coef = 1.0 / (math.sqrt(2 * math.pi) * epsilon)
    comp_pdf = coef * torch.exp(-0.5 * (diff / epsilon) ** 2)  # (G, M)
    return comp_pdf.mean(dim=1)             # (G,)


# ---------------------------------------------------------------------------
# Utility: learned density from the trained flow
# ---------------------------------------------------------------------------

def learned_pdf_from_flow(flow, anchors_exp, x_grid, device):
    anchors_batch = anchors_exp.unsqueeze(0).to(device)  # (1, M, 1)
    x_eval = x_grid.view(1, -1, 1).to(device)            # (1, G, 1)

    with torch.no_grad():
        log_p = flow.log_prob_batch(x_eval, anchors_batch)[0]  # (G,)

    return torch.exp(log_p.cpu())  # (G,)


# ---------------------------------------------------------------------------
# Helper: rebuild the model from checkpoint
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(ckpt, device):
    h = ckpt["hyperparams"]

    set_encoder = SetEncoder(in_dim=h["anchor_dim"],
                             hidden_dim=h["hidden_dim"],
                             z_dim=h["z_dim"]).to(device)

    base_head = BaseHead(z_dim=h["z_dim"],
                         hidden_dim=h["hidden_dim"]).to(device)

    flow = ConditionalFlow1D(
        set_encoder=set_encoder,
        base_head=base_head,
        n_layers=h["n_layers"],
        K=h["k_bins"],
        B=h["bound"],
        z_dim=h["z_dim"],
        hidden_dim=h["hidden_dim"],
        device=str(device),
    ).to(device)

    flow.load_state_dict(ckpt["model_state_dict"])
    flow.eval()
    return flow


# ---------------------------------------------------------------------------
# Main visualization logic
# ---------------------------------------------------------------------------

def main(args, model_path):
    

    # Decide device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load checkpoint and reconstruct model
    
    ckpt = torch.load(os.path.join(exp_dir, model_path), map_location=device, weights_only=False)
    flow = build_model_from_checkpoint(ckpt, device)

    print(f"Loaded model from {model_path}")
    print(f"Best validation NLL = {ckpt.get('best_val_loss', 'N/A')}")

    # Load anchors
    data = torch.load(args.anchors_path, map_location="cpu", weights_only=False)
    anchors_all = data["samples"] if isinstance(data, dict) else data  # (N, 100, 1)
    N = anchors_all.size(0)
    print(f"Loaded anchors of shape {anchors_all.shape}")

    means = data["means"]      # shape: (num_calls, 4)
    stds = data["stds"]        # shape: (num_calls, 4)
    means_np = means.numpy()
    stds_np = stds.numpy()

    # Loop over selected experiment IDs
    for exp_id in args.exp_ids:
        if exp_id < 0 or exp_id >= N:
            print(f"Skipping exp_id={exp_id}: out of bounds.")
            continue

        anchors_exp = anchors_all[exp_id]  # (100, 1)

        # Build an x-grid for this experiment
        eps = args.epsilon
        min_a = anchors_exp.min().item()
        max_a = anchors_exp.max().item()

        x_min = min_a - 3 * eps
        x_max = max_a + 3 * eps
        if x_min == x_max:
            x_min, x_max = min_a - 1.0, max_a + 1.0

        x_grid = torch.linspace(x_min, x_max, args.n_grid_points)

        # Compute true and learned pdfs
        underlying_pdf = mixture_pdf(x_grid.numpy(), means_np[exp_id], stds_np[exp_id])
        target_pdf = mixture_target_pdf(x_grid, anchors_exp, eps)
        learned_pdf = learned_pdf_from_flow(flow, anchors_exp, x_grid, device)

        # # Normalize for good visual comparison
        # target_pdf /= target_pdf.trapezoid(x_grid).clamp_min(1e-10)
        # learned_pdf /= learned_pdf.trapezoid(x_grid).clamp_min(1e-10)

        # Convert to numpy for plotting
        xx = x_grid.numpy()
        tt = target_pdf.numpy()
        ll = learned_pdf.numpy()

        # ---- Plot ----
        plt.figure(figsize=(8, 4))
        plt.plot(xx, underlying_pdf, label="Underlying 4-comp mixture", color="gray", linestyle=":")
        plt.plot(xx, tt, label="Target", linewidth=2)
        plt.plot(xx, ll, "--", label="Learned", linewidth=2)

        # Optional: show anchors as small vertical lines
        anchors_np = anchors_exp.squeeze(-1).numpy()
        for a in anchors_np:
            plt.axvline(a, ymin=0.0, ymax=0.05, alpha=0.2, linewidth=0.7)

        plt.title(f"Experiment {exp_id}: Target vs Learned Distribution")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f"test_Z{z_dim}_experiment_{exp_id}.png"), dpi=300)
        plt.close()


    # -----------------------------------------------------------------------
    # Estimate KL divergence on the test set
    # -----------------------------------------------------------------------

    flow.eval()
    epsilon = args.epsilon  # or set explicitly to match data generation

    @torch.no_grad()
    def log_p_true(x: torch.Tensor, anchors: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        x:       (B, S, 1)
        anchors: (B, M, 1)
        returns: log p_true(x | anchors): (B, S)
        """
        B, S, _ = x.shape
        _, M, _ = anchors.shape

        x_exp = x.squeeze(-1).unsqueeze(-1)           # (B, S, 1)
        a_exp = anchors.squeeze(-1).unsqueeze(1)      # (B, 1, M)

        # (B, S, M) squared distances
        diff2 = (x_exp - a_exp) ** 2

        var = epsilon ** 2
        log_norm = -0.5 * math.log(2.0 * math.pi * var)

        # log N(x | a_j, eps^2) for each j
        log_gauss = log_norm - 0.5 * diff2 / var      # (B, S, M)

        # log (1/M * sum_j exp(log_gauss))
        log_probs = log_gauss.logsumexp(dim=-1) - math.log(M)  # (B, S)
        return log_probs

    @torch.no_grad()
    def estimate_kl_pq(flow, anchors_batch, epsilon, n_samples_per_exp=512):
        """
        Estimates KL( p_true || q_theta ) using Monte Carlo.
        """
        device = anchors_batch.device
        # 1) sample x ~ p_true(.|anchors)
        x_batch = sample_from_anchor_mixture_batch(
            anchors_batch=anchors_batch,
            n_samples=n_samples_per_exp,
            epsilon=epsilon,
        )  # (B, S, 1)

        # 2) log p_true(x|a)
        log_p = log_p_true(x_batch, anchors_batch, epsilon)      # (B, S)

        # 3) log q_theta(x|a)
        log_q = flow.log_prob_batch(x_batch, anchors_batch)      # (B, S)

        # 4) Monte Carlo estimate of KL(p||q)
        kl = (log_p - log_q).mean().item()
        return kl

    # Example usage on full test set
    all_kls = []
    test_dataset = TensorDataset(anchors_all)
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        drop_last=False,
    )
    for (anchors_batch,) in test_loader:
        anchors_batch = anchors_batch.to(device)
        kl_batch = estimate_kl_pq(flow, anchors_batch, epsilon,
                                  n_samples_per_exp=512)
        all_kls.append(kl_batch)

    mean_kl = sum(all_kls) / max(len(all_kls), 1)
    print(f"Estimated KL(p_true || q_theta) on test set: {mean_kl:.6f}")
    return mean_kl


if __name__ == "__main__":
    args = parse_args()
    exp_dir = f"results_{args.epsilon}/"
    z_dim = args.z_dim
    print(f"Testing model with z_dim={z_dim}...")
    model_path = f"best_conditional_flow_Z{z_dim}.pt"
    main(args, model_path)
    
