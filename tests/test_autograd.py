"""Unit tests for autograd differential operators."""

import pytest
import torch
from src.autograd.diff_ops import (
    gradient,
    nth_derivative,
    laplacian,
    divergence,
    directional_derivative,
    hessian_diagonal,
)


def test_gradient_1d():
    """Verify 1st derivative of u(x) = x^3 - 4x^2 + 5x - 2."""
    x = torch.linspace(-3.0, 3.0, 50, requires_grad=True).unsqueeze(-1)
    u = x**3 - 4 * x**2 + 5 * x - 2

    du_dx = gradient(u, x)
    expected = 3 * x**2 - 8 * x + 5

    assert torch.allclose(du_dx, expected, atol=1e-5)


def test_nth_derivative_trig():
    """Verify 1st, 2nd, and 3rd derivatives of u(x) = sin(2x)."""
    x = torch.linspace(0.0, 2 * 3.14159, 100, requires_grad=True).unsqueeze(-1)
    u = torch.sin(2 * x)

    d1 = nth_derivative(u, x, n=1)
    d2 = nth_derivative(u, x, n=2)
    d3 = nth_derivative(u, x, n=3)

    assert torch.allclose(d1, 2 * torch.cos(2 * x), atol=1e-4)
    assert torch.allclose(d2, -4 * torch.sin(2 * x), atol=1e-4)
    assert torch.allclose(d3, -8 * torch.cos(2 * x), atol=1e-4)


def test_laplacian_2d():
    r"""Verify 2D Laplacian: u(x, y) = x^3 + y^4 -> \Delta u = 6x + 12y^2."""
    coords = torch.rand(40, 2, requires_grad=True)
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    u = x**3 + y**4

    lap = laplacian(u, coords)
    expected = 6 * x + 12 * (y**2)

    assert torch.allclose(lap, expected, atol=1e-5)


def test_divergence_2d():
    """Verify 2D divergence of F(x, y) = [x^2 * y, y^3]."""
    coords = torch.rand(30, 2, requires_grad=True)
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    F = torch.cat([x**2 * y, y**3], dim=-1)

    div = divergence(F, coords)
    expected = 2 * x * y + 3 * (y**2)

    assert torch.allclose(div, expected, atol=1e-5)


def test_hessian_diagonal():
    """Verify pure second derivatives: u(x, y) = x^4 + 3*y^3."""
    coords = torch.rand(25, 2, requires_grad=True)
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    u = x**4 + 3 * y**3

    h_diag = hessian_diagonal(u, coords)
    expected = torch.cat([12 * x**2, 18 * y], dim=-1)

    assert torch.allclose(h_diag, expected, atol=1e-5)


def test_backpropagation_through_derivative_into_weights():
    """Verify that a loss computed on a derivative backpropagates into network weights."""
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 1),
    )
    x = torch.linspace(-1.0, 1.0, 20, requires_grad=True).unsqueeze(-1)
    u = model(x)

    # Compute d2u/dx2
    d2u_dx2 = nth_derivative(u, x, n=2, create_graph=True)

    # Residual loss: d2u/dx2 + u = 0
    loss = torch.mean((d2u_dx2 + u)**2)
    loss.backward()

    # Verify that model weights received non-zero gradients
    for name, param in model.named_parameters():
        assert param.grad is not None
        assert not torch.all(param.grad == 0.0), f"Parameter {name} did not receive gradients!"
