"""Unit tests for neural network architectures and quantum models."""

import pytest
import torch
from src.models.mlp import PINNMLP, FourierFeatureEmbedding
from src.models.quantum_net import QuantumPINN


def test_fourier_feature_embedding():
    """Verify Fourier feature output dimension and gradient flow."""
    emb = FourierFeatureEmbedding(in_features=2, num_features=16, scale=2.0)
    x = torch.randn(10, 2, requires_grad=True)
    out = emb(x)

    assert out.shape == (10, 32)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


@pytest.mark.parametrize("activation", ["tanh", "silu", "gelu", "sin"])
def test_pinn_mlp_activations(activation):
    """Verify forward pass across all supported smooth activations."""
    model = PINNMLP(
        in_dim=1,
        out_dim=1,
        hidden_dims=[32, 32],
        activation=activation,
        use_fourier_features=False,
    )
    x = torch.linspace(-1.0, 1.0, 20).unsqueeze(-1)
    out = model(x)
    assert out.shape == (20, 1)


def test_quantum_pinn_energy_param():
    """Verify QuantumPINN creates a trainable energy parameter E."""
    model = QuantumPINN(in_dim=1, E_init=0.5)
    assert isinstance(model.E, torch.nn.Parameter)
    assert abs(model.get_energy() - 0.5) < 1e-6

    # Test gradient flow to E
    x = torch.linspace(-2.0, 2.0, 25).unsqueeze(-1)
    psi = model(x)
    dummy_loss = ((psi - model.E) ** 2).mean()
    dummy_loss.backward()
    assert model.E.grad is not None
    assert model.E.grad.item() != 0.0


def test_quantum_pinn_hard_bc_finite_well():
    """Verify finite well ansatz strictly zeroes wavefunction at boundaries."""
    x_min, x_max = -2.5, 2.5
    model = QuantumPINN(
        in_dim=1,
        ansatz_type="finite_well",
        domain_bounds=(x_min, x_max),
    )

    # Test exact boundary points
    boundaries = torch.tensor([[x_min], [x_max]], dtype=torch.float32)
    psi_bc = model(boundaries)

    assert torch.allclose(psi_bc, torch.zeros_like(psi_bc), atol=1e-7), (
        "Finite well ansatz must vanish exactly at boundaries"
    )


def test_quantum_pinn_norm_integral():
    """Verify compute_norm calculates positive definite integral."""
    model = QuantumPINN(in_dim=1, ansatz_type="gaussian")
    x_grid = torch.linspace(-5.0, 5.0, 200).unsqueeze(-1)
    norm = model.compute_norm(x_grid)

    assert norm.item() > 0.0

