import math

import torch
from torch import nn
import torch.nn.functional as F


def recu(x: torch.Tensor) -> torch.Tensor:
    cubic = x**3 / 3.0
    linear = x - 2.0 / 3.0
    return torch.where(x < 0.0, torch.zeros_like(x), torch.where(x < 1.0, cubic, linear))


def srequ(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2 - F.relu(x - 0.5) ** 2


def get_activation(name: str):
    if name == "recu":
        return recu
    if name == "srequ":
        return srequ
    if hasattr(F, name):
        return getattr(F, name)
    if hasattr(torch, name):
        return getattr(torch, name)
    raise ValueError(f"Unknown activation '{name}'")


class MLP(nn.Module):
    def __init__(self, dim: int, units: list[int], activation: str) -> None:
        super().__init__()
        layer_sizes = [dim] + units
        self.layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(units))]
        )
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
        return self.layers[-1](h)


class PotentialResMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        units: list[int],
        activation: str,
        n_pot: int,
        alpha: float,
        param_dim: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.param_dim = param_dim
        self.alpha = alpha
        self.mlp = MLP(dim + param_dim, units + [n_pot], activation)
        self.gamma_layer = nn.Linear(dim + param_dim, n_pot, bias=False)

    def forward(self, x: torch.Tensor, args: torch.Tensor | None = None) -> torch.Tensor:
        if self.param_dim > 0:
            if args is None:
                raise ValueError("args is required when param_dim > 0")
            x = torch.cat([x, args[:, : self.param_dim]], dim=-1)
        phi = self.mlp(x)
        gamma = self.gamma_layer(x)
        combined = phi + gamma
        output = 0.5 * torch.sum(combined**2, dim=-1)
        output = output + self.alpha * torch.sum(x**2, dim=-1)
        return output


class DissipationMatrixMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        units: list[int],
        activation: str,
        alpha: float,
        is_bounded: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.is_bounded = is_bounded
        self.mlp = MLP(dim, units + [dim * dim], activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        L = self.mlp(x).reshape(batch_size, self.dim, self.dim)
        if self.is_bounded:
            L = torch.tanh(L)
        eye = torch.eye(self.dim, device=x.device, dtype=x.dtype).unsqueeze(0)
        return self.alpha * eye + L @ L.transpose(-1, -2)


class ConservationMatrixMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        units: list[int],
        activation: str,
        is_bounded: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.is_bounded = is_bounded
        self.mlp = MLP(dim, units + [dim * dim], activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        L = self.mlp(x).reshape(batch_size, self.dim, self.dim)
        if self.is_bounded:
            L = torch.tanh(L)
        return L - L.transpose(-1, -2)


class DiffusionDiagonalConstant(nn.Module):
    def __init__(self, dim: int, alpha: float) -> None:
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        lim = 1.0 / math.sqrt(dim)
        weight = torch.empty(dim).uniform_(-lim, lim)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor, args: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.shape[0]
        sigma_diag = torch.sqrt(self.alpha + self.weight**2)
        sigma = sigma_diag.unsqueeze(0).expand(batch_size, -1)
        return torch.diag_embed(sigma)


class DiffusionDiagonal(nn.Module):
    def __init__(
        self,
        dim: int,
        units: list[int],
        activation: str,
        alpha: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.mlp = MLP(dim, units + [dim], activation)

    def forward(self, x: torch.Tensor, args: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.shape[0]
        raw = self.mlp(x)
        sigma_diag = torch.sqrt(self.alpha + raw**2)
        return torch.diag_embed(sigma_diag.reshape(batch_size, self.dim))


class OnsagerNetSDE(nn.Module):
    def __init__(
        self,
        potential: nn.Module,
        dissipation: nn.Module,
        conservation: nn.Module,
        diffusion: nn.Module,
    ) -> None:
        super().__init__()
        self.potential = potential
        self.dissipation = dissipation
        self.conservation = conservation
        self.diffusion_func = diffusion

    def drift(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        args: torch.Tensor | None = None,
        create_graph: bool = True,
    ) -> torch.Tensor:
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            potential = self.potential(x_in, args)
            grad = torch.autograd.grad(
                potential.sum(), x_in, create_graph=create_graph
            )[0]
        M = self.dissipation(x_in)
        W = self.conservation(x_in)
        drift = -(M + W) @ grad.unsqueeze(-1)
        return drift.squeeze(-1)

    def diffusion(
        self, t: torch.Tensor, x: torch.Tensor, args: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.diffusion_func(x, args)


class DriftMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        units: list[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = MLP(dim, units + [dim], activation)

    def forward(self, x: torch.Tensor, args: torch.Tensor | None = None) -> torch.Tensor:
        return self.mlp(x)


class DriftMLPSDE(nn.Module):
    def __init__(
        self,
        drift: nn.Module,
        diffusion: nn.Module,
    ) -> None:
        super().__init__()
        self.drift_func = drift
        self.diffusion_func = diffusion

    def drift(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        args: torch.Tensor | None = None,
        create_graph: bool = True,
    ) -> torch.Tensor:
        if create_graph:
            return self.drift_func(x)
        with torch.no_grad():
            return self.drift_func(x)

    def diffusion(
        self, t: torch.Tensor, x: torch.Tensor, args: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.diffusion_func(x, args)
