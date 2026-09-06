"""General-purpose Physics-Informed Neural Network Trainer."""

from typing import Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn
from tqdm import tqdm


class PINNTrainer:
    """Trainer supporting 2-stage Adam + L-BFGS optimization and dynamic loss balancing."""

    def __init__(
        self,
        model: nn.Module,
        loss_balancer=None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        self.device = device or (
            torch.device("mps")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.model = model.to(self.device)
        self.loss_balancer = loss_balancer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.history: Dict[str, List[float]] = {
            "total_loss": [],
            "residual_loss": [],
        }

    def train_adam(
        self,
        compute_losses_fn: Callable[[nn.Module], Dict[str, torch.Tensor]],
        epochs: int = 2000,
        log_interval: int = 200,
        show_pbar: bool = True,
    ) -> Dict[str, List[float]]:
        """Run Adam optimization loop."""
        self.model.train()
        iterator = range(epochs)
        if show_pbar:
            iterator = tqdm(iterator, desc="Adam Training")

        for epoch in iterator:
            self.optimizer.zero_grad()

            losses = compute_losses_fn(self.model)

            if self.loss_balancer is not None:
                # Optional adaptive update
                self.loss_balancer.update(self.model, losses)
                total_loss = self.loss_balancer.compute_total_loss(losses)
            else:
                # Default: sum all loss components
                total_loss = sum(losses.values())

            total_loss.backward()
            self.optimizer.step()

            # Record history
            self.history["total_loss"].append(total_loss.item())
            for k, v in losses.items():
                hist_key = f"{k}_loss"
                if hist_key not in self.history:
                    self.history[hist_key] = []
                self.history[hist_key].append(v.item())

            if show_pbar and (epoch % log_interval == 0 or epoch == epochs - 1):
                iterator.set_postfix({
                    "Loss": f"{total_loss.item():.4e}",
                    "Res": f"{losses.get('residual', total_loss).item():.4e}",
                })

        return self.history

    def train_lbfgs(
        self,
        compute_losses_fn: Callable[[nn.Module], Dict[str, torch.Tensor]],
        max_iter: int = 500,
        lr: float = 1.0,
    ) -> Dict[str, List[float]]:
        """Run L-BFGS second-stage optimizer for quadratic convergence."""
        self.model.train()
        lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=lr,
            max_iter=max_iter,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        pbar = tqdm(total=max_iter, desc="L-BFGS Fine-Tuning")

        def closure():
            lbfgs.zero_grad()
            losses = compute_losses_fn(self.model)
            if self.loss_balancer is not None:
                total_loss = self.loss_balancer.compute_total_loss(losses)
            else:
                total_loss = sum(losses.values())

            total_loss.backward()

            pbar.update(1)
            pbar.set_postfix({"Loss": f"{total_loss.item():.4e}"})

            self.history["total_loss"].append(total_loss.item())
            for k, v in losses.items():
                hist_key = f"{k}_loss"
                if hist_key not in self.history:
                    self.history[hist_key] = []
                self.history[hist_key].append(v.item())

            return total_loss

        lbfgs.step(closure)
        pbar.close()
        return self.history

