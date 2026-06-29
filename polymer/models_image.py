import math
from typing import Optional

import torch
import torch.nn as nn


class _InducedSetAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_inducing: int,
        ff_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(num_inducing, dim))
        self.attn1 = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(dim)
        self.attn2 = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, dim),
        )
        self.ln3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch = x.shape[0]
        inducing = self.inducing.unsqueeze(0).expand(batch, -1, -1)
        attn_out, _ = self.attn1(
            inducing,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        inducing = self.ln1(inducing + attn_out)
        attn_out, _ = self.attn2(
            x,
            inducing,
            inducing,
            need_weights=False,
        )
        h = self.ln2(x + attn_out)
        ff_out = self.ff(h)
        h = self.ln3(h + ff_out)
        return h


class _PoolingByMultiheadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int, ff_hidden_dim: int, dropout: float):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(num_seeds, dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch = x.shape[0]
        seed = self.seed.unsqueeze(0).expand(batch, -1, -1)
        attn_out, _ = self.attn(
            seed,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = self.ln1(seed + attn_out)
        ff_out = self.ff(out)
        out = self.ln2(out + ff_out)
        return out

class SetEncoderND(nn.Module):
    """
    DeepSets-style encoder with pooling over per-particle features,
    or a Set Transformer encoder when encoder_type="set_transformer".

    You pass a set of features `x`:
      - x: (E, M, D)
      - mask (optional): (E, M) bool, True for valid (non-padding) elements

    Pooling is:
        h_pool = pool_i phi(x_i)
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        z_dim: int = 4,
        pool: str = "mean",
        eps: float = 1e-8,
        encoder_type: str = "deepset",
        num_heads: int = 4,
        num_layers: int = 2,
        num_inducing: int = 16,
        num_seeds: int = 1,
        ff_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.pool = pool
        self.eps = eps

        if encoder_type == "deepset":
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
        elif encoder_type == "set_transformer":
            if hidden_dim % num_heads != 0:
                raise ValueError("hidden_dim must be divisible by num_heads for Set Transformer.")
            if ff_hidden_dim is None:
                ff_hidden_dim = hidden_dim * 2
            self.input_proj = nn.Linear(in_dim, hidden_dim)
            self.sab_layers = nn.ModuleList(
                [
                    _InducedSetAttentionBlock(
                        hidden_dim,
                        num_heads,
                        num_inducing,
                        ff_hidden_dim,
                        dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.pma = _PoolingByMultiheadAttention(
                hidden_dim,
                num_heads,
                num_seeds,
                ff_hidden_dim,
                dropout,
            )
            self.out_proj = nn.Linear(hidden_dim, z_dim)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)
            single = True
        else:
            single = False

        E, M, D = x.shape
        if mask is None:
            mask = torch.ones((E, M), dtype=torch.bool, device=x.device)
        else:
            mask = mask.to(dtype=torch.bool, device=x.device)
        if self.encoder_type == "deepset":
            h = self.phi(x.view(E * M, D)).view(E, M, -1)

            mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
            if self.pool == "mean":
                denom = mask_f.sum(dim=1).clamp_min(self.eps)
                pooled = (h * mask_f).sum(dim=1) / denom
            elif self.pool == "sum":
                pooled = (h * mask_f).sum(dim=1)
            elif self.pool == "max":
                h_masked = h.masked_fill(~mask.unsqueeze(-1), torch.finfo(h.dtype).min)
                pooled = h_masked.max(dim=1).values
            else:
                raise ValueError(f"Unknown pool type: {self.pool}")

            z = self.rho(pooled)
        else:
            key_padding_mask = ~mask
            h = self.input_proj(x)
            for layer in self.sab_layers:
                h = layer(h, key_padding_mask=key_padding_mask)
            pooled = self.pma(h, key_padding_mask=key_padding_mask)
            pooled = pooled.mean(dim=1)
            z = self.out_proj(pooled)

        if single:
            return z.squeeze(0)
        return z


class AnchorGaussianMixtureND:
    """
    Batched ND target distribution with optional padding mask.

    anchors: (E, M, D)
    mask:    (E, M) bool for valid anchors (optional)
    epsilon: isotropic Gaussian std for each component
    """
    def __init__(
        self,
        anchors: torch.Tensor,
        epsilon: float,
        mask: Optional[torch.Tensor] = None,
    ):
        if anchors.dim() == 2:
            anchors = anchors.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)
            self._single = True
        else:
            self._single = False

        if anchors.dim() != 3:
            raise ValueError(f"anchors must be (E,M,D) or (M,D); got {tuple(anchors.shape)}")

        self.anchors = anchors
        self.epsilon = float(epsilon)
        self.mask = mask

    def _effective_mask(self) -> torch.Tensor:
        E, M, _ = self.anchors.shape
        if self.mask is None:
            return torch.ones((E, M), dtype=torch.bool, device=self.anchors.device)
        return self.mask.to(dtype=torch.bool, device=self.anchors.device)

    def sample(self, num_samples: int) -> torch.Tensor:
        anchors = self.anchors
        device = anchors.device
        dtype = anchors.dtype
        E, M, D = anchors.shape

        mask = self._effective_mask()
        weights = mask.to(dtype=torch.float32)
        mass = weights.sum(dim=1, keepdim=True)
        weights = torch.where(mass > 0, weights, torch.ones_like(weights))

        idx = torch.multinomial(weights, num_samples=num_samples, replacement=True)
        gathered = anchors.gather(1, idx.unsqueeze(-1).expand(E, num_samples, D))
        noise = torch.randn(E, num_samples, D, device=device, dtype=dtype)
        samples = gathered + self.epsilon * noise

        if self._single:
            return samples.squeeze(0)
        return samples

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        anchors = self.anchors
        if x.dim() == 2:
            x = x.unsqueeze(0)
            single = True
        else:
            single = False

        if anchors.dim() == 2:
            anchors = anchors.unsqueeze(0)

        E, M, D = anchors.shape
        if x.shape[-1] != D:
            raise ValueError(f"x has dim {x.shape[-1]}, expected {D}")

        mask = self._effective_mask()
        counts = mask.sum(dim=1).clamp_min(1).to(dtype=x.dtype)
        log_weight = -torch.log(counts).view(E, 1, 1)

        var = self.epsilon ** 2
        log_norm = D * math.log(2 * math.pi * var)

        x_exp = x.unsqueeze(2)
        means = anchors.unsqueeze(1)
        diff = x_exp - means
        sqdist = (diff ** 2).sum(dim=-1)
        log_comp = -0.5 * (sqdist / var + log_norm)
        log_probs = log_comp + log_weight
        log_probs = log_probs.masked_fill(~mask.unsqueeze(1), float("-inf"))
        log_p = torch.logsumexp(log_probs, dim=-1)

        if single:
            return log_p.squeeze(0)
        return log_p
