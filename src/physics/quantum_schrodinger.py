r"""Quantum Mechanics: Time-Independent Schrödinger Equation (TISE) Formulations.

Provides potential functions, PDE residual evaluators, normalization constraints,
and orthogonality losses for eigenvalue discovery in 1D and 2D quantum systems.
"""

from typing import Callable, Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
from src.autograd.diff_ops import gradient, nth_derivative, laplacian


class Potential1D:
    """Base class for 1D potential energy functions V(x)."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class HarmonicOscillatorPotential(Potential1D):
    r"""Quantum Harmonic Oscillator: V(x) = 0.5 * m * \omega^2 * x^2.

    In natural units (\hbar = 1, m = 1, \omega = 1): V(x) = 0.5 * x^2.
    Exact eigenvalues: E_n = \hbar \omega (n + 1/2) = 0.5, 1.5, 2.5, ...
    """

    def __init__(self, omega: float = 1.0, m: float = 1.0):
        self.omega = omega
        self.m = m

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.m * (self.omega ** 2) * (x ** 2)


class FiniteSquareWellPotential(Potential1D):
    """Finite square well potential: V(x) = 0 for |x| <= a, V_0 elsewhere."""

    def __init__(self, a: float = 1.0, V0: float = 10.0):
        self.a = a
        self.V0 = V0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        inside = torch.abs(x) <= self.a
        V = torch.full_like(x, self.V0)
        V[inside] = 0.0
        return V


class DoubleWellPotential(Potential1D):
    r"""Symmetric Double-Well potential: V(x) = \lambda * (x^2 - a^2)^2.

    Exhibits quantum tunneling and parity-based energy level splitting
    between symmetric ground state \psi_0 and antisymmetric first excited state \psi_1.
    """

    def __init__(self, a: float = 1.5, lam: float = 1.0):
        self.a = a
        self.lam = lam

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.lam * ((x ** 2 - self.a ** 2) ** 2)


class Schrodinger1D:
    r"""1D Time-Independent Schrödinger Equation Solver:

        -\frac{\hbar^2}{2m} \frac{d^2\psi}{dx^2} + V(x)\psi(x) = E \psi(x)

    In natural units (\hbar = 1, m = 1):
        -\frac{1}{2} \psi''(x) + (V(x) - E)\psi(x) = 0
    """

    def __init__(
        self,
        potential: Potential1D,
        x_range: Tuple[float, float] = (-5.0, 5.0),
        hbar: float = 1.0,
        m: float = 1.0,
    ):
        self.potential = potential
        self.x_min, self.x_max = x_range
        self.hbar = hbar
        self.m = m
        self.kinetic_coeff = (hbar ** 2) / (2.0 * m)

    def compute_residual(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        r"""Compute TISE residual: r(x) = - (hbar^2 / 2m) * psi''(x) + (V(x) - E) * psi(x).

        Args:
            model: QuantumPINN instance with method forward(x) and attribute E.
            x: Collocation coordinates of shape (N, 1). Must have requires_grad=True.

        Returns:
            Residual tensor of shape (N, 1).
        """
        if not x.requires_grad:
            x.requires_grad_(True)

        psi = model(x)
        d2psi_dx2 = nth_derivative(psi, x, n=2, create_graph=True, retain_graph=True)

        V_x = self.potential(x)
        E = model.E

        # Hamiltonian action: H psi - E psi
        residual = -self.kinetic_coeff * d2psi_dx2 + (V_x - E) * psi
        return residual

    def compute_normalization_loss(
        self,
        model: nn.Module,
        x_dense: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute L_norm = ( \int |\psi(x)|^2 dx - 1 )^2.

        Enforcing normalization prevents the trivial collapse to \psi(x) \equiv 0.
        """
        psi = model(x_dense)
        psi_sq = (psi.squeeze(-1)) ** 2
        integral = torch.trapezoid(psi_sq, x_dense.squeeze(-1))
        return (integral - 1.0) ** 2

    def compute_orthogonality_loss(
        self,
        current_model: nn.Module,
        prior_models: List[nn.Module],
        x_dense: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute Gram-Schmidt orthogonality loss: \sum_k ( \int \psi_k(x) \psi_n(x) dx )^2.

        Ensures that excited states are orthogonal to all previously discovered eigenstates.
        """
        if not prior_models:
            return torch.tensor(0.0, device=x_dense.device)

        psi_curr = current_model(x_dense).squeeze(-1)
        x_flat = x_dense.squeeze(-1)

        ortho_loss = torch.tensor(0.0, device=x_dense.device)
        for prior_net in prior_models:
            with torch.no_grad():
                psi_prior = prior_net(x_dense).squeeze(-1)
            # Inner product <psi_prior | psi_curr>
            overlap = torch.trapezoid(psi_prior * psi_curr, x_flat)
            ortho_loss = ortho_loss + (overlap ** 2)

        return ortho_loss

    def sample_domain(
        self,
        n_collocation: int = 1000,
        n_dense: int = 500,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample training collocation points and a dense grid for quadrature."""
        # Random uniform collocation points
        x_colloc = (
            torch.rand(n_collocation, 1) * (self.x_max - self.x_min) + self.x_min
        ).to(device)
        x_colloc.requires_grad_(True)

        # Sorted dense grid for trapezoidal integration
        x_dense = torch.linspace(self.x_min, self.x_max, n_dense).unsqueeze(-1).to(device)

        return x_colloc, x_dense


class Schrodinger2D:
    r"""2D Time-Independent Schrödinger Equation:

        -\frac{1}{2} \nabla^2 \psi(x, y) + V(x, y)\psi(x, y) = E \psi(x, y)
    """

    def __init__(
        self,
        potential_fn: Callable[[torch.Tensor], torch.Tensor],
        bounds: Tuple[float, float, float, float] = (-4.0, 4.0, -4.0, 4.0),
    ):
        self.potential_fn = potential_fn
        self.x_min, self.x_max, self.y_min, self.y_max = bounds

    def compute_residual(self, model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
        """Compute 2D TISE residual."""
        if not xy.requires_grad:
            xy.requires_grad_(True)

        psi = model(xy)
        lap_psi = laplacian(psi, xy, create_graph=True, retain_graph=True)
        V_xy = self.potential_fn(xy)
        E = model.E

        residual = -0.5 * lap_psi + (V_xy - E) * psi
        return residual

