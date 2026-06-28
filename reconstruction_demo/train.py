import math
import argparse
import copy
from typing import Tuple
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ============================================================
# 1. Utility: sample from the true conditional target
# ============================================================

def sample_from_anchor_mixture_batch(
    anchors_batch: torch.Tensor,
    n_samples: int,
    epsilon: float,
) -> torch.Tensor:
    """
    Sample x from the *true* target distribution for a batch of experiments.

    For each experiment (one row of anchors_batch):
      - anchors_batch[b] has shape (M, 1) with M anchors
      - target distribution is a mixture of M Gaussians:
          p(x | anchors) = (1/M) sum_j N(x | anchors_j, epsilon^2)

    Args:
        anchors_batch: (B, M, 1)
        n_samples:     number of x samples per experiment
        epsilon:       std of all local Gaussians

    Returns:
        x_samples: (B, n_samples, 1)
    """
    device = anchors_batch.device
    B, M, _ = anchors_batch.shape

    # Choose which anchor each sample comes from, uniformly at random
    comp_ids = torch.randint(low=0, high=M, size=(B, n_samples), device=device)  # (B, S)

    # Flatten anchors to (B, M) to gather by index
    anchors_flat = anchors_batch.squeeze(-1)  # (B, M)

    # Means for chosen anchor indices: (B, S)
    means = anchors_flat.gather(dim=1, index=comp_ids)

    # Add Gaussian noise
    noise = epsilon * torch.randn(B, n_samples, device=device)
    x = means + noise  # (B, S)

    return x.unsqueeze(-1)  # (B, S, 1)


# ============================================================
# 2. DeepSets encoder: anchor set -> vector z
# ============================================================

class SetEncoder(nn.Module):
    """
    Permutation-invariant encoder on a set of 1D anchors.

    Given anchors S = {a_1, ..., a_M}, we do:
       z = rho( mean_{a in S} phi(a) )

    Here everything is vectorized over the batch of experiments.
    """
    def __init__(self, in_dim: int = 1, hidden_dim: int = 32, z_dim: int = 2):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, anchors: torch.Tensor) -> torch.Tensor:
        """
        anchors: (B, M, in_dim)

        Returns:
            z: (B, z_dim)
        """
        if anchors.dim() == 2:
            anchors = anchors.unsqueeze(-1)  # (B, M, 1)

        # Apply phi to each anchor:
        h = self.phi(anchors)           # (B, M, H)

        # Aggregate over the set dimension M:
        h_agg = h.mean(dim=1)           # (B, H)

        # Map to the final embedding:
        z = self.rho(h_agg)             # (B, z_dim)
        return z


# ============================================================
# 3. BaseHead: z -> Gaussian base parameters
# ============================================================

class BaseHead(nn.Module):
    """
    Same as your original: given z, predict (mu, log_std) for the 1D base Gaussian.
    """
    def __init__(self, z_dim: int = 2, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: (B, z_dim)

        Returns:
            mu:      (B,)
            log_std: (B,)
        """
        out = self.net(z)                      # (B, 2)
        mu = out[..., 0]                       # (B,)
        log_std = out[..., 1].clamp(-5.0, 5.0) # clamp to avoid extreme std
        return mu, log_std


# ============================================================
# 4. FlowHead: z -> RQS spline parameters
# ============================================================

class FlowHead(nn.Module):
    """
    For each z (one experiment), outputs:
      - theta_w: (K,)
      - theta_h: (K,)
      - theta_d: (K-1,)
    """
    def __init__(self, z_dim: int = 2, hidden_dim: int = 32, K: int = 8):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * K - 1),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: (B, z_dim)

        Returns:
            theta_w: (B, K)
            theta_h: (B, K)
            theta_d: (B, K-1)
        """
        K = self.K
        raw = self.net(z)                # (B, 3K-1)
        theta_w = raw[..., 0:K]
        theta_h = raw[..., K:2*K]
        theta_d = raw[..., 2*K:3*K-1]
        return theta_w, theta_h, theta_d


# ============================================================
# 5. Conditional RQS layer (batched)
# ============================================================

class ConditionalRQS1D(nn.Module):
    """
    One 1D RQS transform whose spline parameters are produced by FlowHead(Z).

    x has shape  (B, S, 1)  (batch of experiments * samples)
    z_ctx has shape (B, z_dim)
    """
    def __init__(self, K: int = 8, B: float = 4.0, z_dim: int = 2,
                 hidden_dim: int = 32, device: str = "cpu"):
        super().__init__()
        self.K = K
        self.B = B
        self.device = torch.device(device)
        self.head = FlowHead(z_dim=z_dim, hidden_dim=hidden_dim, K=K)

    def _knots_and_slopes(self, theta_w: torch.Tensor,
                          theta_h: torch.Tensor,
                          theta_d: torch.Tensor):
        """
        theta_w, theta_h: (B, K)
        theta_d:          (B, K-1)

        Returns:
            xk, yk, delta: (B, K+1)
        """
        B_val = self.B
        device = self.device

        widths  = torch.softmax(theta_w, dim=-1) * (2 * B_val)   # (B, K)
        heights = torch.softmax(theta_h, dim=-1) * (2 * B_val)   # (B, K)

        base_shape = theta_w.shape[:-1]  # (B,)

        xk0 = torch.full(base_shape + (1,), -B_val,
                         device=device, dtype=theta_w.dtype)
        xk_rest = -B_val + torch.cumsum(widths, dim=-1)
        xk = torch.cat([xk0, xk_rest], dim=-1)                   # (B, K+1)

        yk0 = torch.full(base_shape + (1,), -B_val,
                         device=device, dtype=theta_h.dtype)
        yk_rest = -B_val + torch.cumsum(heights, dim=-1)
        yk = torch.cat([yk0, yk_rest], dim=-1)                   # (B, K+1)

        s = torch.nn.functional.softplus(theta_d) + 1e-3         # (B, K-1)
        ones = torch.ones(base_shape + (1,), device=device,
                          dtype=theta_d.dtype)
        delta = torch.cat([ones, s, ones], dim=-1)               # (B, K+1)

        return xk, yk, delta

    def forward(self, x: torch.Tensor,
                z_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched RQS forward pass.

        Args:
            x:     (B, S, 1)  input samples
            z_ctx: (B, z_dim) context vectors for each experiment

        Returns:
            y:      (B, S, 1) transformed samples
            logdet: (B, S)    log |dy/dx|
        """
        B_val = self.B
        x_bs = x.squeeze(-1)  # (B, S)

        theta_w, theta_h, theta_d = self.head(z_ctx)             # (B, K), ...
        xk, yk, delta = self._knots_and_slopes(theta_w, theta_h, theta_d)  # (B, K+1)

        B, S = x_bs.shape

        y = x_bs.clone()
        log_det = torch.zeros_like(x_bs)

        left = x_bs < -B_val
        right = x_bs > B_val
        mid = (~left) & (~right)  # inside [-B, B]

        xk_left  = xk[..., :-1]   # (B, K)
        xk_right = xk[..., 1:]    # (B, K)
        yk_left  = yk[..., :-1]
        yk_right = yk[..., 1:]
        dk0_all  = delta[..., :-1]
        dk1_all  = delta[..., 1:]

        x_exp        = x_bs.unsqueeze(-1)        # (B, S, 1)
        xk_left_exp  = xk_left.unsqueeze(1)      # (B, 1, K)
        xk_right_exp = xk_right.unsqueeze(1)     # (B, 1, K)

        cond_left  = x_exp >= xk_left_exp
        cond_right = x_exp <  xk_right_exp
        seg_mask   = cond_left & cond_right      # (B, S, K)

        seg_mask_f = seg_mask.to(x_bs.dtype)

        xk0 = torch.sum(xk_left_exp * seg_mask_f, dim=-1)          # (B, S)
        xk1 = torch.sum(xk_right_exp * seg_mask_f, dim=-1)         # (B, S)
        yk0 = torch.sum(yk_left.unsqueeze(1) * seg_mask_f, dim=-1) # (B, S)
        yk1 = torch.sum(yk_right.unsqueeze(1) * seg_mask_f, dim=-1)
        dk0 = torch.sum(dk0_all.unsqueeze(1) * seg_mask_f, dim=-1)
        dk1 = torch.sum(dk1_all.unsqueeze(1) * seg_mask_f, dim=-1)

        eps = 1e-12
        sk = (yk1 - yk0) / (xk1 - xk0 + eps)
        xi = (x_bs - xk0) / (xk1 - xk0 + eps)
        xi = xi.clamp(0.0, 1.0)
        one_minus_xi = 1.0 - xi

        numer = sk * xi * xi + dk0 * xi * one_minus_xi
        denom = sk + (dk1 + dk0 - 2 * sk) * xi * one_minus_xi
        yh = yk0 + (yk1 - yk0) * (numer / (denom + eps))

        num_deriv = sk * sk * (
            dk1 * xi * xi
            + 2 * sk * xi * one_minus_xi
            + dk0 * one_minus_xi * one_minus_xi
        )
        den_deriv = (sk + (dk1 + dk0 - 2 * sk) * xi * one_minus_xi) ** 2
        dydx = num_deriv / (den_deriv + eps)

        y = torch.where(mid, yh, y)
        log_det = torch.where(mid, torch.log(dydx + eps), log_det)

        return y.unsqueeze(-1), log_det   # (B, S, 1), (B, S)


# ============================================================
# 6. Conditional flow on top of RQS layers
# ============================================================

class ConditionalFlow1D(nn.Module):
    """
    Conditional flow: anchors (B, M, 1) -> Z via SetEncoder, etc.
    """
    def __init__(self,
                 set_encoder: SetEncoder,
                 base_head: BaseHead,
                 n_layers: int = 2,
                 K: int = 16,
                 B: float = 10.0,
                 z_dim: int = 2,
                 hidden_dim: int = 32,
                 device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.set_encoder = set_encoder
        self.base_head = base_head
        self.layers = nn.ModuleList(
            [
                ConditionalRQS1D(
                    K=K, B=B, z_dim=z_dim, hidden_dim=hidden_dim,
                    device=self.device
                )
                for _ in range(n_layers)
            ]
        )

    def _base_log_prob(self,
                       z_latent: torch.Tensor,
                       z_ctx: torch.Tensor) -> torch.Tensor:
        """
        Base log-density:

        Args:
            z_latent: (B, S, 1)
            z_ctx:    (B, z_dim)

        Returns:
            log p_base(z_latent | z_ctx): (B, S)
        """
        mu, log_std = self.base_head(z_ctx)  # (B,), (B,)
        std = log_std.exp()

        u = z_latent.squeeze(-1)   # (B, S)

        mu = mu.unsqueeze(1)       # (B, 1)
        log_std = log_std.unsqueeze(1)
        std = std.unsqueeze(1)
        var = std ** 2

        return -0.5 * (
            (u - mu) ** 2 / var
            + math.log(2.0 * math.pi)
            + 2.0 * log_std
        )

    def log_prob_batch(self,
                       x: torch.Tensor,
                       anchors: torch.Tensor) -> torch.Tensor:
        """
        Fully batched log p_theta(x | anchors).

        Args:
            x:       (B, S, 1) samples for each experiment
            anchors: (B, M, 1) anchors for each experiment

        Returns:
            log p(x | anchors): (B, S)
        """
        x = x.to(self.device)
        anchors = anchors.to(self.device)

        z_ctx = self.set_encoder(anchors)        # (B, z_dim)

        h = x
        Bsz, S, _ = x.shape
        log_det = torch.zeros(Bsz, S, device=self.device)

        for layer in self.layers:
            h, ld = layer(h, z_ctx)             # h: (B, S, 1), ld: (B, S)
            log_det += ld

        log_p_base = self._base_log_prob(h, z_ctx)   # (B, S)

        return log_p_base + log_det                  # (B, S)


# ============================================================
# 7. Training loop with argparse + best-model saving
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    # >>> NEW: hyperparameters from command line
    parser.add_argument("--anchor_dim", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_samples_per_exp", type=int, default=512, 
                        help="Number of x samples to draw per experiment in each training batch for computing KL divergence.")

    parser.add_argument("--z_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--k_bins", type=int, default=50)
    parser.add_argument("--bound", type=float, default=4.0)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=8000)
    parser.add_argument("--val_fraction", type=float, default=0.1)


    parser.add_argument("--device", type=int, default=1)

    parser.add_argument("--input_data_path", type=str, default="data_generation/")

    return parser.parse_args()


def main():
    args = parse_args()

    # Decide device
    if args.device >= 0:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # ----------------------------------------
    # 1) Anchors: shape (N_EXPERIMENTS, N_ANCHORS, 1)
    #    Here we just generate random anchors; in your project you can
    #    replace this with loading your saved [10000, 100, 1] tensor.
    # ----------------------------------------
        
    train_data = torch.load(os.path.join(f"{args.input_data_path}", "gaussian_mixture_data_train.pt"), map_location=device, weights_only=False)
    train_anchors = train_data["samples"].to(device)  # (N, M, 1)
    print(f"Anchors shape: {train_anchors.shape}")

    
    val_data = torch.load(os.path.join(f"{args.input_data_path}", "gaussian_mixture_data_test.pt"), map_location=device, weights_only=False)
    val_anchors = val_data["samples"].to(device)  # (N, M, 1)
    print(f"Validation Anchors shape: {val_anchors.shape}")



    train_dataset = TensorDataset(train_anchors)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_dataset = TensorDataset(val_anchors)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    

    # ----------------------------------------
    # 2) Model setup
    # ----------------------------------------
    set_encoder = SetEncoder(
        in_dim=args.anchor_dim,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
    ).to(device)

    base_head = BaseHead(
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    flow = ConditionalFlow1D(
        set_encoder=set_encoder,
        base_head=base_head,
        n_layers=args.n_layers,
        K=args.k_bins,
        B=args.bound,
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        device=str(device),
    ).to(device)

    optimizer = optim.Adam(flow.parameters(), lr=args.lr)

    # >>> NEW: track best validation loss and save best model
    best_val_loss = float("inf")
    best_state_dict = None

    # ----------------------------------------
    # 3) Training
    # ----------------------------------------
    for epoch in tqdm(range(1, args.n_epochs + 1), desc="Training epochs"):
        flow.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch_idx, (anchors_batch,) in enumerate(train_loader):
            anchors_batch = anchors_batch.to(device)  # (B, M, 1)

            # Sample from the *true* mixture defined by these anchors
            x_batch = sample_from_anchor_mixture_batch(
                anchors_batch=anchors_batch,
                n_samples=args.n_samples_per_exp,
                epsilon=args.epsilon,
            )  # (B, S, 1)

            # Compute log p_theta(x | anchors) under the conditional flow
            log_px = flow.log_prob_batch(x_batch, anchors_batch)  # (B, S)

            # Negative log-likelihood loss
            loss = -log_px.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # Validation
        flow.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for (anchors_batch,) in val_loader:
                anchors_batch = anchors_batch.to(device)
                x_batch = sample_from_anchor_mixture_batch(
                    anchors_batch=anchors_batch,
                    n_samples=args.n_samples_per_exp,
                    epsilon=args.epsilon,
                )
                log_px = flow.log_prob_batch(x_batch, anchors_batch)
                loss = -log_px.mean()

                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        

        if epoch % 500 == 0 or epoch == 1 or epoch == args.n_epochs:
            print(
                f"Epoch [{epoch}/{args.n_epochs}] "
                f"train NLL: {avg_train_loss:.4f} "
                f"val NLL: {avg_val_loss:.4f}"
            )

        # >>> NEW: update best model if validation improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = copy.deepcopy(flow.state_dict())
            # print(f"  -> New best model (val NLL = {best_val_loss:.4f})")

    # ----------------------------------------
    # 4) Save best model checkpoint
    # ----------------------------------------
    if best_state_dict is None:
        best_state_dict = flow.state_dict()

    # Move params to CPU for portability
    best_state_cpu = {k: v.cpu() for k, v in best_state_dict.items()}

    checkpoint = {
        "model_state_dict": best_state_cpu,
        "hyperparams": vars(args),          # >>> NEW: save all args
        "best_val_loss": best_val_loss,
    }
    save_dir = f"results_{args.epsilon}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_conditional_flow_Z{args.z_dim}.pt")
    torch.save(checkpoint, save_path)
    print(f"Best model saved to {save_path} (val NLL = {best_val_loss:.4f})")


if __name__ == "__main__":
    main()
