"""Quantum Neural Network for solving the Time-Independent Schrödinger Equation (TISE).

Encapsulates both the wavefunction network psi(x) and the learnable energy
eigenvalue E, with options for hard boundary condition ansatzes.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from src.models.mlp import PINNMLP


class QuantumPINN(nn.Module):
    r"""Neural architecture for quantum eigenvalue discovery.

    In the TISE:
        \hat{H} \psi(x) = \left( -\frac{\hbar^2}{2m} \frac{d^2}{dx^2} + V(x) \right) \psi(x) = E \psi(x)

    Here, both the eigenstate \psi(x) and the energy eigenvalue E are unknowns!
    QuantumPINN models \psi(x) via an MLP and represents E as a trainable scalar
    `torch.nn.Parameter`.

    Boundary Condition Ansatz Options:
    1. 'gaussian': \psi(x) = \exp(-x^2 / 2) \cdot \mathcal{N}(x).
       Guarantees exponential decay at \pm \infty (ideal for Quantum Harmonic Oscillator).
    2. 'finite_well': \psi(x) = (x - x_min)(x_max - x) \cdot \mathcal{N}(x).
       Enforces exact zero Dirichlet boundary conditions at the well edges [x_min, x_max].
    3. 'none': \psi(x) = \mathcal{N}(x). Soft boundary enforcement via penalty loss.
    """

    def __init__(
        self,
        in_dim: int = 1,
        hidden_dims: List[int] = [64, 64, 64],
        activation: str = "tanh",
        ansatz_type: str = "gaussian",
        domain_bounds: Optional[Tuple[float, float]] = None,
        E_init: float = 1.0,
        use_fourier_features: bool = False,
        fourier_dim: int = 16,
        fourier_scale: float = 1.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.ansatz_type = ansatz_type.lower()
        self.domain_bounds = domain_bounds

        # Learnable energy eigenvalue scalar parameter E
        self.E = nn.Parameter(torch.tensor(float(E_init), dtype=torch.float32))

        # Core neural network predicting the envelope/amplitude
        self.net = PINNMLP(
            in_dim=in_dim,
            out_dim=1,
            hidden_dims=hidden_dims,
            activation=activation,
            use_fourier_features=use_fourier_features,
            fourier_dim=fourier_dim,
            fourier_scale=fourier_scale,
            dropout_rate=dropout_rate,
        )

        if self.ansatz_type == "finite_well":
            if domain_bounds is None:
                raise ValueError("domain_bounds=(x_min, x_max) is required for 'finite_well' ansatz.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting wavefunction psi(x).

        Args:
            x: Spatial coordinate tensor of shape (N, in_dim).

        Returns:
            psi: Wavefunction values of shape (N, 1).
        """
        raw_out = self.net(x)

        if self.ansatz_type == "gaussian":
            # Radial distance squared r^2 = sum_i x_i^2
            r2 = torch.sum(x**2, dim=-1, keepdim=True)
            decay = torch.exp(-0.5 * r2)
            return decay * raw_out

        elif self.ansatz_type == "finite_well":
            x_min, x_max = self.domain_bounds
            # For 1D: (x - x_min) * (x_max - x)
            x_coord = x[..., 0:1]
            envelope = (x_coord - x_min) * (x_max - x_coord)
            return envelope * raw_out

        elif self.ansatz_type == "none":
            return raw_out

        else:
            raise ValueError(f"Unknown ansatz_type: {self.ansatz_type}")

    def get_energy(self) -> float:
        """Return the current learned energy eigenvalue as a Python float."""
        return self.E.detach().item()

    def compute_norm(self, x_grid: torch.Tensor) -> torch.Tensor:
        r"""Compute the normalization integral \int |\psi(x)|^2 dx using trapezoidal rule.

        Args:
            x_grid: 1D grid tensor of shape (N, 1), sorted in ascending order.

        Returns:
            Scalar tensor representing \int |\psi|^2 dx.
        """
        psi = self.forward(x_grid)
        psi_sq = psi.squeeze(-1) ** 2  # (N,)
        x_flat = x_grid.squeeze(-1)    # (N,)
        integral = torch.trapezoid(psi_sq, x_flat)
        return integral

