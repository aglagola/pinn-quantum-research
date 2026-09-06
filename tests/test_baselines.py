"""Unit tests for analytical baselines and finite-difference / Numerov solvers."""

import numpy as np
import pytest
from src.baselines.analytical import (
    heat_1d_analytical,
    qho_analytical_energy,
    qho_analytical_wavefunction,
)
from src.baselines.numerov_fd import solve_tise_finite_difference


def test_heat_1d_analytical():
    """Verify heat equation analytical solution properties."""
    x = np.linspace(0.0, 1.0, 50)
    t0 = np.zeros_like(x)
    u0 = heat_1d_analytical(x, t0, alpha=0.4, L=1.0, k=1)

    # At t=0, u(x, 0) = sin(pi*x)
    assert np.allclose(u0, np.sin(np.pi * x), atol=1e-5)

    # At t > 0, solution decays
    t1 = np.ones_like(x) * 0.5
    u1 = heat_1d_analytical(x, t1, alpha=0.4, L=1.0, k=1)
    assert np.all(u1 < u0 + 1e-6)
    assert u1[0] == pytest.approx(0.0, abs=1e-6)
    assert u1[-1] == pytest.approx(0.0, abs=1e-6)


def test_qho_exact_energies():
    """Verify QHO analytical energy spectrum: E_n = n + 0.5."""
    for n in range(5):
        assert qho_analytical_energy(n) == pytest.approx(n + 0.5)


def test_qho_analytical_normalization():
    """Verify analytical QHO wavefunctions have unit L2 norm."""
    x = np.linspace(-6.0, 6.0, 2000)
    for n in range(3):
        psi = qho_analytical_wavefunction(x, n=n)
        norm = np.trapezoid(psi ** 2, x)
        assert norm == pytest.approx(1.0, rel=1e-3)


def test_finite_difference_tise_harmonic_oscillator():
    """Verify FD matrix solver computes QHO eigenvalues to high precision."""
    pot_fn = lambda x: 0.5 * (x ** 2)
    x_grid, evals, evecs = solve_tise_finite_difference(
        potential_fn=pot_fn,
        x_range=(-6.0, 6.0),
        n_points=1200,
        num_states=4,
    )

    expected_evals = [0.5, 1.5, 2.5, 3.5]
    for i, exp in enumerate(expected_evals):
        # Numerical error should be well under 1e-3 with 1200 grid points
        assert evals[i] == pytest.approx(exp, abs=1e-3)

    # Verify orthonormality: <psi_0 | psi_1> = 0
    overlap = np.trapezoid(evecs[:, 0] * evecs[:, 1], x_grid)
    assert overlap == pytest.approx(0.0, abs=1e-4)

