#!/usr/bin/env python

import torch

class KATLayer(torch.nn.Module):
    def __init__(self, input_dim : int, output_dim : int, 
                 xrange : tuple[float, float] = (0, 1), n : int = 10, order : int = 1, std_w : float = 0.001, 
                 dropout = None):
        super(KATLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n = n
        self.dropout = dropout
        self.order = order
        self.normal = torch.distributions.Normal(0, 1)
        
        if xrange[1] <= xrange[0]:
            raise ValueError("Invalid range. Upper bound must be greater than lower bound.")
        sigma_init = (xrange[1] - xrange[0]) / n / 3
        
        mx_start = torch.linspace(xrange[0], xrange[1], n).view(1, 1, 1, n)
        self.register_buffer('mx_start', mx_start)

        self.mx_train = torch.nn.Parameter(torch.full((input_dim, output_dim), 0.0))

        self.scale = torch.nn.Parameter(torch.full((input_dim, output_dim), 1.0))
    
        self.sigma = torch.nn.Parameter(torch.full((input_dim, output_dim, n), sigma_init))

        self.alpha = torch.nn.Parameter(torch.full((input_dim, output_dim, n), 0.0))
        
        self.w = torch.nn.Parameter(torch.normal(mean=0.0, std=std_w, size=(input_dim, output_dim, n)))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(-1).unsqueeze(-1)
        mx_train_expanded = self.mx_train.unsqueeze(-1)
        scale_expanded = self.scale.unsqueeze(-1)

        z = (x_expanded - torch.abs(scale_expanded) * self.mx_start - mx_train_expanded) / (torch.abs(self.sigma) + 1e-8)
        
        if self.order > 1:
            f = torch.exp(-z**(2*self.order)) * 2 * self.normal.cdf(self.alpha * z)
        else:
            f = torch.exp(-z**2) * 2 * self.normal.cdf(self.alpha * z)

        if self.dropout is not None:
            f = self.dropout(f)
        
        return torch.sum(self.w * f, dim=(1, 3))
    
    def eval_func(self, x : float, i : int, j : int) -> float:
        if i < 0 or i >= self.input_dim:
            raise ValueError(f"Input index i={i} is out of bounds (0-{self.input_dim-1})")
        if j < 0 or j >= self.output_dim:
            raise ValueError(f"Output index j={j} is out of bounds (0-{self.output_dim-1})")

        with torch.no_grad():
            x_tensor = torch.full((self.n,), x, device=self.mx_train.device)
            mx_start = self.mx_start.view(self.n)
            mx_train = self.mx_train[i,j].expand(self.n)
            scale = self.scale[i,j].expand(self.n)

            z = (x_tensor - torch.abs(scale) * mx_start - mx_train) / (torch.abs(self.sigma[i,j,:]) + 1e-8)
        
            if self.order > 1:
                f = torch.exp(-z**(2*self.order)) * 2 * self.normal.cdf(self.alpha[i,j,:] * z)
            else:
                f = torch.exp(-z**2) * 2 * self.normal.cdf(self.alpha[i,j,:] * z)

            return torch.sum(self.w[i,j,:] * f).item()
    
class KATMLP(torch.nn.Module):
    def __init__(self, input_size : int, hidden_sizes : list[int], output_size : int,
                 xranges : list[tuple[float, float], tuple[float, float]] = [(0, 1), (-1, 1)], n : int = 10, 
                 order : int = 1, std_w : float = 0.001, dropout = None):
        super(KATMLP, self).__init__()
        layers = []

        layers.append(KATLayer(input_size, hidden_sizes[0], xranges[0], n, order, std_w, dropout))

        for i in range(1, len(hidden_sizes)):
            layers.append(KATLayer(hidden_sizes[i-1], hidden_sizes[i], xranges[1] if len(xranges) == 2 else xranges[0], n, order, std_w, dropout))

        layers.append(KATLayer(hidden_sizes[-1], output_size, xranges[1] if len(xranges) == 2 else xranges[0], n, order, std_w, dropout))

        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.model(x)