"""Unit tests for physics residual computations."""

import pytest
import torch
from src.models.mlp import PINNMLP
from src.models.quantum_net import QuantumPINN
from src.physics.classical_pde import HeatEquation1D
from src.physics.quantum_schrodinger import (
    HarmonicOscillatorPotential,
    Schrodinger1D,
)


def test_heat_equation_residual_shape():
    """Verify heat equation residual computes (N, 1) tensor."""
    pde = HeatEquation1D(alpha=0.4, L=1.0, T=1.0)
    model = PINNMLP(in_dim=2, out_dim=1, hidden_dims=[32, 32], activation="tanh")

    xt = torch.rand(40, 2, requires_grad=True)
    residual = pde.compute_residual(model, xt)

    assert residual.shape == (40, 1)
    assert residual.requires_grad


def test_schrodinger_residual_and_norm():
    """Verify 1D TISE residual and normalization loss calculations."""
    pot = HarmonicOscillatorPotential(omega=1.0)
    tise = Schrodinger1D(potential=pot, x_range=(-5.0, 5.0))
    model = QuantumPINN(in_dim=1, ansatz_type="gaussian", E_init=0.5)

    x_colloc, x_dense = tise.sample_domain(n_collocation=50, n_dense=100)

    residual = tise.compute_residual(model, x_colloc)
    assert residual.shape == (50, 1)

    norm_loss = tise.compute_normalization_loss(model, x_dense)
    assert norm_loss.ndim == 0
    assert norm_loss.item() >= 0.0


def test_schrodinger_orthogonality_loss():
    """Verify orthogonality loss returns zero for empty priors and positive for identical nets."""
    pot = HarmonicOscillatorPotential(omega=1.0)
    tise = Schrodinger1D(potential=pot)
    model1 = QuantumPINN(in_dim=1, ansatz_type="gaussian", E_init=0.5)
    model2 = QuantumPINN(in_dim=1, ansatz_type="gaussian", E_init=1.5)

    _, x_dense = tise.sample_domain(n_dense=100)

    # Empty priors -> 0 loss
    zero_ortho = tise.compute_orthogonality_loss(model2, [], x_dense)
    assert zero_ortho.item() == 0.0

    # Same model evaluated against itself -> non-zero positive overlap
    self_ortho = tise.compute_orthogonality_loss(model1, [model1], x_dense)
    assert self_ortho.item() > 0.0
