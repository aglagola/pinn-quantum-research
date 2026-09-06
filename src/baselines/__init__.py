"""Analytical and numerical baselines for benchmarking PINN solutions."""

from src.baselines.analytical import (
    heat_1d_analytical,
    qho_analytical_wavefunction,
    qho_analytical_energy,
)
from src.baselines.numerov_fd import solve_tise_finite_difference, numerov_solve

__all__ = [
    "heat_1d_analytical",
    "qho_analytical_wavefunction",
    "qho_analytical_energy",
    "solve_tise_finite_difference",
    "numerov_solve",
]

