import os
import random
import numpy as np
import torch
import torch.nn as nn


# Set the random seed for reproduction
def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Autoencoder Model
# ---------------------------------------------------------------------------

class MLPEncoder(nn.Module):
    """
    MLP Encoder that takes flattened set input and produces a fixed-size latent code.
    
    Input: (batch_size, M * D) where M is number of points, D is dimension per point
    Output: (batch_size, z_dim)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, z_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, M * D)
        returns: (batch_size, z_dim)
        """
        return self.net(x)


class MLPDecoder(nn.Module):
    """
    MLP Decoder that takes latent code and reconstructs the flattened set.
    
    Input: (batch_size, z_dim)
    Output: (batch_size, M * D)
    """
    def __init__(self, z_dim: int, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch_size, z_dim)
        returns: (batch_size, M * D)
        """
        return self.net(z)


class Autoencoder(nn.Module):
    """
    Simple MLP-based Autoencoder.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, z_dim: int = 16):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dim, z_dim)
        self.decoder = MLPDecoder(z_dim, hidden_dim, input_dim)
        self.z_dim = z_dim
        self.input_dim = input_dim
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, M, D) or (batch_size, M * D)
        returns: (z, x_reconstructed)
            z: (batch_size, z_dim)
            x_reconstructed: (batch_size, M, D) (same shape as input)
        """
        # Flatten if input is 3D
        original_shape = x.shape
        if x.dim() == 3:
            batch_size = x.shape[0]
            x_flat = x.view(batch_size, -1)
        else:
            x_flat = x
        
        z = self.encoder(x_flat)
        x_recon_flat = self.decoder(z)
        
        # Reshape back to original shape
        x_recon = x_recon_flat.view(original_shape)
        return z, x_recon


# ---------------------------------------------------------------------------
# DeepSet-based Autoencoder Model
# ---------------------------------------------------------------------------

class DeepSetEncoder(nn.Module):
    """
    DeepSet Encoder with permutation-invariant pooling.
    
    Input: (batch_size, M, D) where M is number of points, D is dimension per point
    Output: (batch_size, z_dim)
    
    Architecture:
        phi: point-wise transformation (M, D) -> (M, hidden_dim)
        pool: permutation-invariant aggregation (mean/max/sum)
        rho: aggregated features -> latent code
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64, z_dim: int = 16, pool: str = "mean"):
        super().__init__()
        self.pool = pool
        
        # Point-wise transformation network (phi)
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
        )
        
        # Aggregation network (rho)
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, z_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, M, D)
        returns: (batch_size, z_dim)
        """
        batch_size, M, D = x.shape
        
        # Point-wise transformation: (B, M, D) -> (B*M, D) -> (B*M, H)
        x_flat = x.view(batch_size * M, D)
        h = self.phi(x_flat)  # (B*M, hidden_dim)
        
        # Reshape back and pool: (B*M, H) -> (B, M, H) -> (B, H)
        H = h.shape[-1]
        h = h.view(batch_size, M, H)
        
        if self.pool == "mean":
            h_pooled = h.mean(dim=1)  # (B, H)
        elif self.pool == "max":
            h_pooled = h.max(dim=1)[0]  # (B, H)
        elif self.pool == "sum":
            h_pooled = h.sum(dim=1)  # (B, H)
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")
        
        # Final transformation to latent code
        z = self.rho(h_pooled)  # (B, z_dim)
        return z


class DeepSetAutoencoder(nn.Module):
    """
    DeepSet-based Autoencoder with permutation-invariant encoder.
    
    Encoder: DeepSet (permutation-invariant)
    Decoder: MLP (reconstructs flattened output)
    """
    def __init__(self, in_dim: int, n_points: int, hidden_dim: int = 256, z_dim: int = 16, pool: str = "mean"):
        super().__init__()
        self.in_dim = in_dim
        self.n_points = n_points
        self.z_dim = z_dim
        self.output_dim = n_points * in_dim
        
        # DeepSet encoder (permutation-invariant)
        self.encoder = DeepSetEncoder(in_dim, hidden_dim, z_dim, pool)
        
        # MLP decoder (reconstructs flattened set)
        self.decoder = MLPDecoder(z_dim, hidden_dim, self.output_dim)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, M, D)
        returns: (z, x_reconstructed)
            z: (batch_size, z_dim)
            x_reconstructed: (batch_size, M, D)
        """
        batch_size, M, D = x.shape
        
        # Encode with DeepSet (permutation-invariant)
        z = self.encoder(x)  # (B, z_dim)
        
        # Decode to flattened output
        x_recon_flat = self.decoder(z)  # (B, M*D)
        
        # Reshape back to (B, M, D)
        x_recon = x_recon_flat.view(batch_size, M, D)
        
        return z, x_recon


# ---------------------------------------------------------------------------
# ODE Dynamics Model
# ---------------------------------------------------------------------------

class OnsagerNet(nn.Module):
    """
    Neural network for ODE right-hand side function.
    Based on Onsager principle with learned potential and dissipation matrix.
    """
    def __init__(
        self, 
        input_dim: int, 
        n_nodes: list = None,
        forcing: bool = True,
        ResNet: bool = True,
        pot_beta: float = 0.1,
        ons_min_d: float = 0.1,
        init_gain: float = 0.1,
        f_act = None,
        f_linear: bool = False,
    ):
        super().__init__()
        
        if n_nodes is None:
            n_nodes = [64, 64, 64]
        
        n_nodes = np.array([input_dim] + n_nodes)
        
        self.nL = n_nodes.size
        self.nVar = n_nodes[0]
        self.nNodes = np.zeros(self.nL + 1, dtype=np.int32)
        self.nNodes[:self.nL] = n_nodes
        self.nNodes[self.nL] = self.nVar ** 2
        self.nPot = self.nVar
        self.forcing = forcing
        self.pot_beta = pot_beta
        self.ons_min_d = ons_min_d
        self.F_act = f_act if f_act is not None else nn.SiLU()
        self.f_linear = f_linear
        
        if ResNet:
            self.ResNet = 1.0
            assert np.sum(n_nodes[1:] - n_nodes[1]) == 0, \
                f'ResNet structure is not implemented for {n_nodes}'
        else:
            self.ResNet = 0.0

        self.baselayer = nn.ModuleList([
            nn.Linear(self.nNodes[i], self.nNodes[i + 1])
            for i in range(self.nL - 1)
        ])
        self.MatLayer = nn.Linear(self.nNodes[self.nL - 1], self.nVar ** 2)
        self.PotLayer = nn.Linear(self.nNodes[self.nL - 1], self.nPot)
        self.PotLinear = nn.Linear(self.nVar, self.nPot)

        # Initialization
        bias_eps = 0.5
        for i in range(self.nL - 1):
            nn.init.xavier_uniform_(self.baselayer[i].weight, gain=init_gain)
            nn.init.uniform_(self.baselayer[i].bias, 0, bias_eps * init_gain)

        nn.init.xavier_uniform_(self.MatLayer.weight, gain=init_gain)
        w = torch.empty(self.nVar, self.nVar, requires_grad=True)
        nn.init.orthogonal_(w, gain=1.0)
        self.MatLayer.bias.data = w.view(-1, self.nVar ** 2)

        nn.init.orthogonal_(self.PotLayer.weight, gain=init_gain)
        nn.init.uniform_(self.PotLayer.bias, 0, init_gain)
        nn.init.orthogonal_(self.PotLinear.weight, gain=init_gain)
        nn.init.uniform_(self.PotLinear.bias, 0, init_gain)

        if self.forcing:
            if self.f_linear:
                self.lforce = nn.Linear(self.nVar, self.nVar)
            else:
                self.lforce = nn.Linear(self.nNodes[self.nL - 1], self.nVar)
            nn.init.orthogonal_(self.lforce.weight, init_gain)
            nn.init.uniform_(self.lforce.bias, 0.0, bias_eps * init_gain)
    
    def makePDM(self, matA):
        """Make Positive Definite Matrix from a given matrix."""
        AL = torch.tril(matA, 0)
        AU = torch.triu(matA, 1)
        Aant = AU - torch.transpose(AU, 1, 2)
        Asym = torch.bmm(AL, torch.transpose(AL, 1, 2))
        return Asym, Aant

    def learn_V(self, inputs):
        """Learn potential function V."""
        output = self.F_act(self.baselayer[0](inputs))
        for i in range(1, self.nL - 1):
            output = self.F_act(self.baselayer[i](output)) + self.ResNet * output
            
        PotLinear = self.PotLinear(inputs)
        Pot = self.PotLayer(output) + PotLinear
        V = torch.sum(Pot ** 2) + self.pot_beta * torch.sum(inputs ** 2)
        return V
            
    def forward(self, inputs):
        shape = inputs.shape
        inputs = inputs.view(-1, self.nVar)
        
        # Compute gradient of potential
        g = torch.func.jacrev(self.learn_V, argnums=0)(inputs)
        g = -g.view(-1, self.nVar, 1)
        
        # Compute dissipation matrix
        output = self.F_act(self.baselayer[0](inputs))
        for i in range(1, self.nL - 1):
            output = self.F_act(self.baselayer[i](output)) + self.ResNet * output

        matA = self.MatLayer(output)
        matA = matA.view(-1, self.nVar, self.nVar)
        AM, AW = self.makePDM(matA)
        MW = AW + AM
        
        # Compute forcing term
        if self.forcing:
            if self.f_linear:
                lforce = self.lforce(inputs)
            else:
                lforce = self.lforce(output)
        
        # Combine terms
        output = torch.matmul(MW, g) + self.ons_min_d * g

        if self.forcing:
            output = output + lforce.view(-1, self.nVar, 1)

        output = output.view(*shape)
        return output


# ---------------------------------------------------------------------------
# DeepSet + ODE Combined Model
# ---------------------------------------------------------------------------

class DeepSetODE(nn.Module):
    """
    Combined model: DeepSet encoder + ODE dynamics.
    
    Input: (batch_size, M, D) set of points
    Encoder: DeepSet -> (batch_size, z_dim) latent code
    Dynamics: ODE model predicts dz/dt in latent space
    """
    def __init__(
        self, 
        in_dim: int, 
        z_dim: int, 
        hidden_dim: int = 256, 
        pool: str = "mean",
        ode_nodes: list = None,
    ):
        super().__init__()
        self.z_dim = z_dim
        
        # DeepSet encoder (permutation-invariant)
        self.encoder = DeepSetEncoder(in_dim, hidden_dim, z_dim, pool)
        
        # ODE dynamics in latent space
        if ode_nodes is None:
            ode_nodes = [64, 64, 64]
        self.dynamics = OnsagerNet(input_dim=z_dim, n_nodes=ode_nodes)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode set to latent space."""
        return self.encoder(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, M, D)
        returns: dz/dt in latent space (batch_size, z_dim)
        """
        z = self.encoder(x)  # (B, z_dim)
        dzdt = self.dynamics(z)  # (B, z_dim)
        return dzdt
