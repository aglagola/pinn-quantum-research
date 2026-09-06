"""Static / fixed loss weighting scheme for PINNs."""

from typing import Dict
import torch


class FixedLossWeights:
    """Fixed constant weighting for multi-objective PINN loss components."""

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

    def compute_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted sum of loss terms."""
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        for key, loss_val in losses.items():
            w = self.weights.get(key, 1.0)
            total = total + w * loss_val
        return total

    def update(self, *args, **kwargs) -> None:
        """No-op for fixed weights."""
        pass

