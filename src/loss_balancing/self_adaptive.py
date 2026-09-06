r"""Self-Adaptive PINN (SAPINN) loss weighting.

Reference:
    McClenny, L., & Braga-Neto, U. (2023).
    "Self-Adaptive Physics-Informed Neural Networks."
    Journal of Computational Physics, 474, 111722.

Theory:
    Associates each collocation point with a trainable weight \lambda_i >= 0.
    The network minimizes loss w.r.t. \theta, while the weights \lambda_i maximize
    the loss w.r.t. \lambda via gradient ascent. This automatically focuses
    computational capacity onto high-residual / stiff regions of the domain.
"""

from typing import Tuple
import torch
import torch.nn as nn


class SelfAdaptiveWeights(nn.Module):
    """Trainable point-wise collocation weights updated via gradient ascent."""

    def __init__(self, num_points: int, init_val: float = 1.0):
        super().__init__()
        # Parameterized as logits to ensure positive weights via sigmoid or softplus
        self.raw_weights = nn.Parameter(torch.full((num_points, 1), float(init_val)))

    @property
    def weights(self) -> torch.Tensor:
        """Return strictly positive adaptive weights."""
        return 2.0 * torch.sigmoid(self.raw_weights)

    def compute_weighted_loss(self, squared_residuals: torch.Tensor) -> torch.Tensor:
        """Compute weighted mean squared residual."""
        return torch.mean(self.weights * squared_residuals)

