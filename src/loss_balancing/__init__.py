"""Loss balancing and adaptive weighting strategies for PINNs."""

from src.loss_balancing.fixed import FixedLossWeights
from src.loss_balancing.lr_annealing import LearningRateAnnealing
from src.loss_balancing.self_adaptive import SelfAdaptiveWeights

__all__ = ["FixedLossWeights", "LearningRateAnnealing", "SelfAdaptiveWeights"]

