r"""Learning Rate Annealing / Dynamic Weighting for PINNs.

Reference:
    Wang, S., Teng, Y., & Perdikaris, P. (2021).
    "Understanding and mitigating gradient pathologies in physics-informed neural networks."
    SIAM Journal on Scientific Computing, 43(5), A3055-A3081.

Theory:
    In standard PINN training, the gradients of the boundary loss often dwarf the
    gradients of the PDE residual loss by several orders of magnitude, causing the
    network to fit trivial boundary conditions while ignoring internal PDE physics.
    Learning Rate Annealing balances gradient magnitudes dynamically via an Exponential
    Moving Average (EMA) of gradient norm ratios.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn


class LearningRateAnnealing:
    r"""Dynamic loss weight balancing based on gradient norms w.r.t. network weights.

    Formula for each auxiliary task k (e.g. boundary, initial condition, normalization):
        \hat{\lambda}_k = \frac{\max_{\theta} |\nabla_\theta \mathcal{L}_{res}|}{\overline{|\nabla_\theta \mathcal{L}_k|}}
        \lambda_k^{(t+1)} = (1 - \alpha) \lambda_k^{(t)} + \alpha \hat{\lambda}_k
    """

    def __init__(
        self,
        aux_loss_keys: List[str],
        primary_key: str = "residual",
        alpha: float = 0.1,
        initial_weights: Optional[Dict[str, float]] = None,
        update_freq: int = 10,
    ):
        self.aux_loss_keys = aux_loss_keys
        self.primary_key = primary_key
        self.alpha = alpha
        self.update_freq = update_freq
        self.step_count = 0

        # Initialize weights
        self.weights: Dict[str, float] = {}
        for key in aux_loss_keys:
            init_val = initial_weights.get(key, 1.0) if initial_weights else 1.0
            self.weights[key] = init_val

    def compute_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute balanced total loss."""
        total = losses[self.primary_key]
        for key in self.aux_loss_keys:
            if key in losses:
                total = total + self.weights[key] * losses[key]
        return total

    def update(self, model: nn.Module, losses: Dict[str, torch.Tensor]) -> None:
        """Update adaptive weights via gradient norm inspection."""
        self.step_count += 1
        if self.step_count % self.update_freq != 0:
            return

        if self.primary_key not in losses:
            return

        # Extract primary gradients w.r.t. model parameters
        primary_loss = losses[self.primary_key]
        params = [p for p in model.parameters() if p.requires_grad]

        primary_grads = torch.autograd.grad(
            primary_loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )

        max_primary_grad = 0.0
        for g in primary_grads:
            if g is not None:
                max_primary_grad = max(max_primary_grad, torch.max(torch.abs(g)).item())

        # Compute gradient norms for each auxiliary loss
        for key in self.aux_loss_keys:
            if key not in losses:
                continue

            aux_loss = losses[key]
            aux_grads = torch.autograd.grad(
                aux_loss,
                params,
                retain_graph=True,
                allow_unused=True,
            )

            mean_aux_grad_sum = 0.0
            count = 0
            for g in aux_grads:
                if g is not None:
                    mean_aux_grad_sum += torch.mean(torch.abs(g)).item()
                    count += 1

            mean_aux_grad = (mean_aux_grad_sum / count) if count > 0 else 1e-8
            if mean_aux_grad < 1e-12:
                mean_aux_grad = 1e-8

            target_lambda = max_primary_grad / mean_aux_grad

            # EMA update
            self.weights[key] = (1.0 - self.alpha) * self.weights[key] + self.alpha * target_lambda

