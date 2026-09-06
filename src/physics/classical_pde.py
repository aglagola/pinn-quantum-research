r"""Classical Partial Differential Equations (Heat and Wave Equations).

Formulates the physics loss, collocation sampling, and analytical solutions
for the 1D Heat equation (diffusion) and 1D Wave equation.
"""

from typing import Dict, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
from src.autograd.diff_ops import gradient, nth_derivative


class HeatEquation1D:
    r"""1D Diffusion / Heat Equation:

        \frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0,
        x \in [0, L], \quad t \in [0, T]

    Boundary Conditions:
        u(0, t) = 0, \quad u(L, t) = 0
    Initial Condition:
        u(x, 0) = \sin\left(\frac{k \pi x}{L}\right)

    Analytical Solution:
        u(x, t) = \sin\left(\frac{k \pi x}{L}\right) \exp\left(-\alpha \left(\frac{k \pi}{L}\right)^2 t\right)
    """

    def __init__(
        self,
        alpha: float = 0.4,
        L: float = 1.0,
        T: float = 1.0,
        k: int = 1,
    ):
        self.alpha = alpha
        self.L = L
        self.T = T
        self.k = k

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the exact analytical solution."""
        spatial_factor = torch.sin((self.k * math.pi * x) / self.L)
        decay_rate = self.alpha * ((self.k * math.pi / self.L) ** 2)
        temporal_factor = torch.exp(-decay_rate * t)
        return spatial_factor * temporal_factor

    def compute_residual(self, model: nn.Module, xt: torch.Tensor) -> torch.Tensor:
        r"""Compute the PDE residual r(x, t) = u_t - \alpha * u_xx.

        Args:
            model: Neural network taking (N, 2) [x, t] and outputting (N, 1) u.
            xt: Tensor of shape (N, 2) where xt[:, 0:1] is x and xt[:, 1:2] is t.
                Must have requires_grad=True.

        Returns:
            Residual tensor of shape (N, 1).
        """
        if not xt.requires_grad:
            xt.requires_grad_(True)

        u = model(xt)

        # 1st gradient [du/dx, du/dt]
        grad_u = gradient(u, xt, create_graph=True, retain_graph=True)
        u_x = grad_u[..., 0:1]
        u_t = grad_u[..., 1:2]

        # 2nd spatial derivative d^2u / dx^2
        grad_ux = gradient(u_x, xt, create_graph=True, retain_graph=True)
        u_xx = grad_ux[..., 0:1]

        # PDE residual
        residual = u_t - self.alpha * u_xx
        return residual

    def sample_data(
        self,
        n_collocation: int = 2000,
        n_initial: int = 200,
        n_boundary: int = 200,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """Sample interior collocation points, initial condition, and boundary points."""
        # 1. Collocation points inside domain (x, t) in [0, L] x [0, T]
        x_f = torch.rand(n_collocation, 1) * self.L
        t_f = torch.rand(n_collocation, 1) * self.T
        xt_f = torch.cat([x_f, t_f], dim=-1).to(device)
        xt_f.requires_grad_(True)

        # 2. Initial condition points at t = 0
        x_0 = torch.rand(n_initial, 1) * self.L
        t_0 = torch.zeros(n_initial, 1)
        xt_0 = torch.cat([x_0, t_0], dim=-1).to(device)
        u_0 = torch.sin((self.k * math.pi * x_0) / self.L).to(device)

        # 3. Boundary points: x = 0 and x = L for t in [0, T]
        n_b_half = n_boundary // 2
        t_b1 = torch.rand(n_b_half, 1) * self.T
        x_b1 = torch.zeros(n_b_half, 1)

        t_b2 = torch.rand(n_b_half, 1) * self.T
        x_b2 = torch.full((n_b_half, 1), self.L)

        xt_b = torch.cat([
            torch.cat([x_b1, t_b1], dim=-1),
            torch.cat([x_b2, t_b2], dim=-1),
        ], dim=0).to(device)
        u_b = torch.zeros(n_boundary, 1).to(device)

        return {
            "xt_f": xt_f,
            "xt_0": xt_0,
            "u_0": u_0,
            "xt_b": xt_b,
            "u_b": u_b,
        }


class WaveEquation1D:
    r"""1D Damped Wave Equation:

        \frac{\partial^2 u}{\partial t^2} + \gamma \frac{\partial u}{\partial t} - c^2 \frac{\partial^2 u}{\partial x^2} = 0
    """

    def __init__(self, c: float = 1.0, gamma: float = 0.1, L: float = 1.0, T: float = 2.0):
        self.c = c
        self.gamma = gamma
        self.L = L
        self.T = T

    def compute_residual(self, model: nn.Module, xt: torch.Tensor) -> torch.Tensor:
        """Compute wave equation residual u_tt + gamma*u_t - c^2*u_xx."""
        if not xt.requires_grad:
            xt.requires_grad_(True)

        u = model(xt)
        grad_u = gradient(u, xt, create_graph=True, retain_graph=True)
        u_x = grad_u[..., 0:1]
        u_t = grad_u[..., 1:2]

        grad_ux = gradient(u_x, xt, create_graph=True, retain_graph=True)
        u_xx = grad_ux[..., 0:1]

        grad_ut = gradient(u_t, xt, create_graph=True, retain_graph=True)
        u_tt = grad_ut[..., 1:2]

        residual = u_tt + self.gamma * u_t - (self.c ** 2) * u_xx
        return residual

