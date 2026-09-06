"""Physics definitions and PDE residual formulations."""

from src.physics.classical_pde import HeatEquation1D, WaveEquation1D
from src.physics.quantum_schrodinger import (
    HarmonicOscillatorPotential,
    FiniteSquareWellPotential,
    DoubleWellPotential,
    Schrodinger1D,
    Schrodinger2D,
)

__all__ = [
    "HeatEquation1D",
    "WaveEquation1D",
    "HarmonicOscillatorPotential",
    "FiniteSquareWellPotential",
    "DoubleWellPotential",
    "Schrodinger1D",
    "Schrodinger2D",
]

