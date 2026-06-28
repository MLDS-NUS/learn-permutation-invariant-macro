import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from tqdm import tqdm


class DiffusitivityNet(nn.Module):
    """Generate diffusitivity term for SDE
    """
    def __init__(self, n_dim, mode='arbitrary'):
        super(DiffusitivityNet, self).__init__()
        self.n_dim = n_dim
        self.mode = mode

        if mode == 'arbitrary':
            self.output_layer = nn.Sequential(
                nn.Linear(n_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_dim * n_dim),
            )

        elif mode in ['diagonal']:
            self.output_layer = nn.Sequential(
                nn.Linear(n_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_dim)
            )

        elif mode == 'constant_diagonal':
            self.kernel = nn.Parameter(torch.ones(n_dim))  # Parameters are automatically considered for optimization
        
        elif mode == 'constant':
            self.kernel = nn.Parameter(torch.ones(n_dim * n_dim))

        else:
            raise ValueError(f"Mode {mode} is unknown")
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        """Diffusion term 
        Args:
            x (Tensor): Input tensor of shape [B, n_dim]

        Returns:
            Tensor: Output tensor of shape [B, n_dim, n_dim]
        """        
        # General case: arbitrary diffusion matrix
        if self.mode == 'arbitrary':

            output = self.output_layer(x)
            output = output.view(-1, self.n_dim, self.n_dim)
        
        # Diagonal diffusion matrix
        elif self.mode == 'diagonal':
            
            output = self.output_layer(x)
            output = torch.diag_embed(output)
            output = output.view(-1, self.n_dim, self.n_dim)

        # Constant diagonal diffusion matrix
        elif self.mode == 'constant_diagonal':
            output = self.kernel.unsqueeze(0).repeat(x.shape[0], 1)
            output = torch.diag_embed(output)
            output = output.view(-1, self.n_dim, self.n_dim)

        # Constant diffusion matrix
        elif self.mode == 'constant':
            output = self.kernel.unsqueeze(0).repeat(x.shape[0], 1)
            output = output.view(-1, self.n_dim, self.n_dim) 

        return output

class SDE_Net(nn.Module):
    def __init__(self, delta_t, mode='constant_diagonal', n_dim=3, epsilon=1e-4):
        super().__init__()
        self.delta_t = nn.Parameter(torch.tensor(delta_t), requires_grad=False)
        self.n_dim = n_dim
        self.mode = mode
        # Drift network
        self.drift_net = nn.Sequential(
            nn.Linear(n_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_dim)
        )
        self.epsilon = epsilon
        
        # Diffusitivity network
        self.sigma_net = DiffusitivityNet(n_dim=n_dim, mode=mode)

        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.uniform_(module.bias, 0, 0.1)
    
    def drift(self, x):
        # x: [B, 3]
        drift = self.drift_net(x)

        return drift
    
    def custom_loss(self, x0, x1, dt=None):
        """Compute the custom loss between two states.
            Loss = (x1 - x0 - drift * dt)^T * (K \Sigma)^{-1} * (x1 - x0 - drift * dt) + log|K \Sigma|
        
        Args:
            x0 (Tensor): The initial state. Shape: [B, n_dim]
            x1 (Tensor): The target state. Shape: [B, n_dim]
            dt (float, optional): The time step size. If None, uses self.delta_t. Defaults to None.

        Returns:
            Tensor: The computed loss.
        """
        if dt == None:
            dt = self.delta_t

        drift = self.drift(x0) # [B, 3]
        sigma = self.sigma_net(x0) # [B, 3, 3]

        cov_matrix = torch.bmm(sigma, sigma.transpose(1, 2)) * dt.view(-1, 1, 1)
        # add small value to the diagonal for numerical stability
        Sigma = cov_matrix + torch.eye(self.n_dim, device=x0.device) * self.epsilon
        Sigma_inv = torch.linalg.inv(Sigma)

        X = x1 - x0 - drift * dt
        a1 = torch.einsum('ij,ijk,ik->i', X, Sigma_inv, X)
        a1 = a1.view(-1, 1)

        a2 = torch.linalg.slogdet(Sigma)[1]  # Returns a tuple (sign, logdet); we need logdet
        a2 = a2.view(-1, 1)

        return torch.mean(a1 + a2)
    
    def predict(self, x, steps, dt=None):
        """Predict the long-term trajectory for a given number of steps.
        
        Args:
            x (Tensor): The initial state. Shape: [B, n_dim]
            steps (int): The number of steps to predict.
            dt (float, optional): The time step size. If None, uses self.delta_t. Defaults to None.
        
        Returns:
            Tensor: The predicted trajectory. Shape: [B, n_dim, steps]
        """
        if dt == None:
            dt = self.delta_t
        
        # x: [B, latent_dim]
        predict_tra = [x]
        for _ in tqdm(range(steps-1)):

            x0 = predict_tra[-1]
            drift = self.drift(x0) # [B, 3]
            sigma = self.sigma_net(x0) # [B, 3, 3]
            delta_W = torch.normal(mean=0., std=torch.sqrt(dt), size=(x.shape[0],self.n_dim), device=x.device)
            x1 = x0 + drift * dt + torch.einsum('ijk,ik->ij', sigma, delta_W)

            predict_tra.append(x1)

        predict_tra = torch.stack(predict_tra, 2).transpose(2, 1)
        return predict_tra
        
    def forward(self, z0, noise):
        """Euler-Maruyama scheme for one step

        Args:
            z0 (Tensor): The initial state.
            noise (Tensor): The noise to be added.

        Returns:
            Tensor: The next state.
        """
        # z0: [B, D]
        # noise: [B, D]
        drift = self.drift(z0) # [B, D]
        sigma = self.sigma_net(z0) # [B, D, D]

        z1 = z0 + drift * self.delta_t + torch.einsum('ijk,ik->ij', sigma, noise) * torch.sqrt(self.delta_t)
        return z1




class OnsagerNet_original(nn.Module):
    """ A neural network to for the rhs function of an ODE,
    used to fitting data """

    def __init__(self, input_dim, n_nodes=[64, 64, 64], forcing=True, ResNet=True,
                 pot_beta=0.1,
                 ons_min_d=0.1,
                 init_gain=0.1,
                 f_act= nn.SiLU(),
                 f_linear=False,
                 ):
        super().__init__()

        n_nodes = np.array([input_dim] + n_nodes)

        if n_nodes is None:   # used for subclasses
            return
        self.nL = n_nodes.size # 2
        self.nVar = n_nodes[0] # 3
        self.nNodes = np.zeros(self.nL+1, dtype=np.int32) # 3
        self.nNodes[:self.nL] = n_nodes  # [3, 20, 9]
        self.nNodes[self.nL] = self.nVar**2 # 9
        self.nPot = self.nVar # 3
        self.forcing = forcing # True
        self.pot_beta = pot_beta # 0.1 
        self.ons_min_d = ons_min_d # 0.1
        self.F_act = f_act
        self.f_linear = f_linear
        if ResNet:
            self.ResNet = 1.0
            assert np.sum(n_nodes[1:]-n_nodes[1]) == 0, \
                f'ResNet structure is not implemented for {n_nodes}'
        else:
            self.ResNet = 0.0

        self.baselayer = nn.ModuleList([nn.Linear(self.nNodes[i], 
                                                  self.nNodes[i+1])
                                        for i in range(self.nL-1)]) # [3->20]
        self.MatLayer = nn.Linear(self.nNodes[self.nL-1], self.nVar**2) # [20->9]
        self.PotLayer = nn.Linear(self.nNodes[self.nL-1], self.nPot) # [20->3]
        self.PotLinear = nn.Linear(self.nVar, self.nPot) # [3->3]

        # Initialization 
        # init baselayer
        bias_eps = 0.5
        for i in range(self.nL-1):
            init.xavier_uniform_(self.baselayer[i].weight, gain=init_gain)
            init.uniform_(self.baselayer[i].bias, 0, bias_eps*init_gain)

        # init MatLayer
        init.xavier_uniform_(self.MatLayer.weight, gain=init_gain)
        w = torch.empty(self.nVar, self.nVar, requires_grad=True)
        nn.init.orthogonal_(w, gain=1.0)
        self.MatLayer.bias.data = w.view(-1, self.nVar**2)

        # init PotLayer and PotLinear
        init.orthogonal_(self.PotLayer.weight, gain=init_gain)
        init.uniform_(self.PotLayer.bias, 0, init_gain)
        init.orthogonal_(self.PotLinear.weight, gain=init_gain)
        init.uniform_(self.PotLinear.bias, 0, init_gain)

        # init lforce
        if self.forcing:
            if self.f_linear:
                self.lforce = nn.Linear(self.nVar, self.nVar) # [3->3]
            else:
                self.lforce = nn.Linear(self.nNodes[self.nL-1], self.nVar)
            init.orthogonal_(self.lforce.weight, init_gain)
            init.uniform_(self.lforce.bias, 0.0, bias_eps*init_gain)
    
    def makePDM(self, matA):
        """ Make Positive Definite Matrix from a given matrix
        matA has a size (batch_size x N x N) """
        AL = torch.tril(matA, 0)
        AU = torch.triu(matA, 1)
        Aant = AU - torch.transpose(AU, 1, 2)
        Asym = torch.bmm(AL, torch.transpose(AL, 1, 2))
        return Asym,  Aant

    def learn_V(self, inputs):
        output = self.F_act(self.baselayer[0](inputs))
        for i in range(1, self.nL-1):
            output = (self.F_act(self.baselayer[i](output))
                        + self.ResNet*output)
            
        PotLinear = self.PotLinear(inputs)
        Pot = self.PotLayer(output) + PotLinear
        V = torch.sum(Pot**2) + self.pot_beta * torch.sum(inputs**2)
        return V
            
    def forward(self, inputs):
        shape = inputs.shape
        inputs = inputs.view(-1, self.nVar) # [B, 3]
        ### change the following lines to use torch.func.jacrev
        # with torch.enable_grad():
        #     inputs.requires_grad_(True)
        #     inputs.retain_grad()
        #     output = self.F_act(self.baselayer[0](inputs))
        #     for i in range(1, self.nL-1):
        #         output = (self.F_act(self.baselayer[i](output))
        #                   + self.ResNet*output)
                
        #     PotLinear = self.PotLinear(inputs)
        #     Pot = self.PotLayer(output) + PotLinear
        #     V = torch.sum(Pot**2) + self.pot_beta * torch.sum(inputs**2)

        #     g, = torch.autograd.grad(V, inputs, create_graph=True)
        #     g = - g.view(-1, self.nVar, 1)

        g = torch.func.jacrev(self.learn_V, argnums=0)(inputs)        
        g = - g.view(-1, self.nVar, 1) # [B, Z_dim, 1]
        

        output = self.F_act(self.baselayer[0](inputs))
        for i in range(1, self.nL-1):
            output = (self.F_act(self.baselayer[i](output))
                      + self.ResNet*output)

        matA = self.MatLayer(output)
        matA = matA.view(-1, self.nVar, self.nVar)
        AM, AW = self.makePDM(matA)
        MW = AW+AM
        
        if self.forcing:
            if self.f_linear:
                lforce = self.lforce(inputs)
            else:
                lforce = self.lforce(output)
        
        output = torch.matmul(MW, g) + self.ons_min_d * g

        if self.forcing:
            output = output + lforce.view(-1, self.nVar, 1)    

        output = output.view(*shape)
        return output




class drift_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.drift_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.uniform_(module.bias, 0, 0.1)
    

    def forward(self, x):
        return self.drift_net(x)