import math
from typing import Tuple

import torch
import torch.nn as nn

import torch.nn.functional as F
import math



# a lightweight MLP
class light_mlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):    
        super().__init__()        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)





class AnchorGaussianMixtureND:
    """
    Batched ND target:
      - anchors: (E, M, D)
      - epsilon: scalar std for isotropic local Gaussians

    For experiment e:
      p_e(x) = (1/M) * sum_j N_D(x | anchors[e, j], epsilon^2 I)
    """
    def __init__(self, anchors: torch.Tensor, epsilon: float):
        """
        anchors: (E, M, D) or (M, D) (in which case E=1)
        """
        self.device = anchors.device
        if anchors.dim() == 2:
            anchors = anchors.unsqueeze(0)   # (1,M,D)
        assert anchors.dim() == 3
        self.anchors = anchors               # (E,M,D)
        self.num_experiments, self.num_components, self.dim = anchors.shape
        self.epsilon = float(epsilon)
        self.log_weight = -math.log(self.num_components)

    def sample(self, batch_size: int, exp_idx: int | None = None) -> torch.Tensor:
        """
        Returns:
          if exp_idx is None: (E,B,D) samples (B per experiment)
          else:               (B,D) for experiment exp_idx
        """
        E, M, D = self.anchors.shape
        device = self.device

        if exp_idx is not None:
            anchors = self.anchors[exp_idx:exp_idx+1]  # (1,M,D)
            E_eff = 1
        else:
            anchors = self.anchors
            E_eff = E

        comp_ids = torch.randint(0, M, (E_eff, batch_size), device=device)  # (E_eff,B)
        eps = torch.randn(E_eff, batch_size, D, device=device)              # (E_eff,B,D)

        exp_ids = torch.arange(E_eff, device=device).unsqueeze(1).expand(E_eff, batch_size)
        selected = anchors[exp_ids, comp_ids]  # (E_eff,B,D)

        x = selected + self.epsilon * eps      # (E_eff,B,D)

        if exp_idx is not None:
            return x.squeeze(0)               # (B,D)
        return x                               # (E,B,D)

    def log_prob(self, x: torch.Tensor, exp_idx: int | None = None) -> torch.Tensor:
        """
        x: (E,B,D) if exp_idx is None, else (B,D) or (1,B,D)
        returns:
          log p(x): (E,B) if exp_idx is None, else (B,)
        """
        anchors = self.anchors
        E, M, D = anchors.shape
        device = self.device
        var = self.epsilon ** 2
        log_norm = D * math.log(2 * math.pi * var)

        if exp_idx is not None:
            anchors = anchors[exp_idx:exp_idx+1]     # (1,M,D)
            if x.dim() == 2:
                x = x.unsqueeze(0)                  # (1,B,D)
        else:
            assert x.dim() == 3, "x must be (E,B,D) when exp_idx is None"

        # x = x.to(device)
        # broadcast over components
        # x: (E,B,1,D), anchors: (E,1,M,D)
        x_exp = x.unsqueeze(2)                      # (E,B,1,D)
        means = anchors.unsqueeze(1)                # (E,1,M,D)
        diff = x_exp - means                        # (E,B,M,D)
        sqdist = (diff ** 2).sum(dim=-1)            # (E,B,M)

        log_comp = -0.5 * (sqdist / var + log_norm) # (E,B,M)
        log_probs = self.log_weight + log_comp      # (E,B,M)
        log_p = torch.logsumexp(log_probs, dim=-1)  # (E,B)

        if exp_idx is not None:
            return log_p.squeeze(0)                 # (B,)
        return log_p







class SetEncoderND(nn.Module):
    """
    Permutation-invariant encoder on sets of D-dim anchors.
    Supports:
      anchors: (E, M, D)  -> Z: (E, z_dim)
      anchors: (M, D)     -> Z: (z_dim,)
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64, z_dim: int = 4, pool: str = "mean"):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, z_dim),
        )
        self.pool = pool

    def forward(self, anchors: torch.Tensor) -> torch.Tensor:
        if anchors.dim() == 2:
            # (M,D) -> single experiment
            anchors = anchors.unsqueeze(0)  # (1,M,D)
            single = True
        else:
            single = False

        E, M, D = anchors.shape
        x = anchors.view(E * M, D)          # (E*M, D)
        h = self.phi(x)                     # (E*M, H)
        H = h.shape[-1]
        if self.pool == "mean":
            h = h.view(E, M, H).mean(dim=1)     # (E, H)
        elif self.pool == "max":
            h = h.view(E, M, H).max(dim=1)[0]  # (E, H)
        elif self.pool == "sum":
            h = h.view(E, M, H).sum(dim=1)     # (E, H)
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")
        
        z = self.rho(h)                     # (E, z_dim)

        if single:
            return z.squeeze(0)
        return z


