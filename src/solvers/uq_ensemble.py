"""Uncertainty Quantification (UQ) via Deep Ensembles and Monte Carlo Dropout."""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class DeepEnsembleUQ:
    """Uncertainty Quantification using an ensemble of M independently trained PINNs."""

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        num_models: int = 5,
        device: Optional[torch.device] = None,
    ):
        self.model_factory = model_factory
        self.num_models = num_models
        self.device = device or (
            torch.device("mps")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.models: List[nn.Module] = []

    def train_ensemble(
        self,
        trainer_fn: Callable[[nn.Module, int], nn.Module],
    ) -> None:
        """Train each ensemble member with different seed and data subsampling."""
        self.models = []
        for m_idx in range(self.num_models):
            torch.manual_seed(1000 + m_idx * 42)
            np.random.seed(1000 + m_idx * 42)

            model = self.model_factory().to(self.device)
            print(f"\n[Ensemble Model {m_idx + 1}/{self.num_models}] Training...")
            trained_model = trainer_fn(model, m_idx)
            self.models.append(trained_model)

    def predict(self, x_grid: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute mean prediction and epistemic standard deviation across models."""
        predictions = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                pred = model(x_grid.to(self.device)).cpu().numpy().flatten()
                predictions.append(pred)

        all_preds = np.stack(predictions, axis=0)  # (num_models, N)
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)

        return mean_pred, std_pred, all_preds


class MCDropoutUQ:
    """Uncertainty Quantification using Monte Carlo Dropout at inference time."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or (
            torch.device("mps")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )

    def predict(
        self,
        x_grid: torch.Tensor,
        num_mc_samples: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform stochastic forward passes with dropout active."""
        # Enable dropout during inference
        self.model.train()

        samples = []
        with torch.no_grad():
            for _ in range(num_mc_samples):
                out = self.model(x_grid.to(self.device)).cpu().numpy().flatten()
                samples.append(out)

        samples_arr = np.stack(samples, axis=0)
        mean_pred = np.mean(samples_arr, axis=0)
        std_pred = np.std(samples_arr, axis=0)

        self.model.eval()
        return mean_pred, std_pred
