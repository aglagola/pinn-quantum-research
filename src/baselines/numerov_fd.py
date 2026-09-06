r"""High-precision numerical solvers for the 1D Schrödinger Equation.

Implements:
1. Tridiagonal Finite Difference Matrix Eigensolver (via scipy.linalg.eigh_tridiagonal).
2. Numerov algorithm for 4th-order ODE integration.
"""

from typing import Callable, Tuple
import numpy as np
import scipy.linalg as la


def solve_tise_finite_difference(
    potential_fn: Callable[[np.ndarray], np.ndarray],
    x_range: Tuple[float, float] = (-6.0, 6.0),
    n_points: int = 1000,
    num_states: int = 5,
    hbar: float = 1.0,
    m: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Solve 1D TISE via Finite Difference matrix diagonalization.

    Hamiltonian discretization:
        H_{i,i} = \frac{\hbar^2}{m \Delta x^2} + V(x_i)
        H_{i, i\pm 1} = -\frac{\hbar^2}{2 m \Delta x^2}

    Args:
        potential_fn: Callable returning V(x) given numpy array x.
        x_range: Domain boundaries (x_min, x_max).
        n_points: Number of spatial discretization intervals.
        num_states: Number of lowest eigenstates to return.
        hbar: Reduced Planck constant.
        m: Particle mass.

    Returns:
        x_grid: Grid points array of shape (N,).
        eigenvalues: Sorted array of shape (num_states,) with energy levels E_0, E_1, ...
        eigenvectors: Normalized wavefunctions array of shape (N, num_states).
    """
    x_min, x_max = x_range
    x = np.linspace(x_min, x_max, n_points)
    dx = x[1] - x[0]

    # Kinetic matrix constants
    coeff = (hbar ** 2) / (2.0 * m * (dx ** 2))

    # Main diagonal: 2 * coeff + V(x)
    V = potential_fn(x)
    diag = 2.0 * coeff + V

    # Off-diagonal: -coeff
    off_diag = np.full(n_points - 1, -coeff)

    # Solve symmetric tridiagonal eigenvalue problem
    # select_range gives indices 0 to num_states - 1
    evals, evecs = la.eigh_tridiagonal(
        diag,
        off_diag,
        select="i",
        select_range=(0, num_states - 1),
    )

    # Normalize wavefunctions such that \int |\psi(x)|^2 dx = 1
    for k in range(num_states):
        norm = np.sqrt(np.trapezoid(evecs[:, k] ** 2, x))
        if norm > 1e-12:
            evecs[:, k] /= norm

        # Enforce positive sign at peak for consistent phase
        peak_idx = np.argmax(np.abs(evecs[:, k]))
        if evecs[peak_idx, k] < 0:
            evecs[:, k] = -evecs[:, k]

    return x, evals, evecs


def numerov_solve(
    potential_fn: Callable[[np.ndarray], np.ndarray],
    E: float,
    x_range: Tuple[float, float] = (-6.0, 6.0),
    n_points: int = 1000,
    hbar: float = 1.0,
    m: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Numerov integration for \psi''(x) = f(x)\psi(x) where f(x) = (2m/\hbar^2)(V(x) - E).

    Numerov recurrence relation:
        \psi_{n+1} = \frac{2(1 - \frac{5}{12} h^2 f_n)\psi_n - (1 + \frac{1}{12} h^2 f_{n-1})\psi_{n-1}}{1 + \frac{1}{12} h^2 f_{n+1}}
    """
    x_min, x_max = x_range
    x = np.linspace(x_min, x_max, n_points)
    h = x[1] - x[0]
    h2 = h ** 2

    # f(x) = 2m/hbar^2 * (V(x) - E)
    V = potential_fn(x)
    f = (2.0 * m / (hbar ** 2)) * (V - E)

    psi = np.zeros(n_points)
    # Asymptotic boundary condition: decaying into classically forbidden zone
    psi[0] = 0.0
    psi[1] = 1e-4

    c = h2 / 12.0
    for i in range(1, n_points - 1):
        num = 2.0 * (1.0 - 5.0 * c * f[i]) * psi[i] - (1.0 + c * f[i - 1]) * psi[i - 1]
        denom = 1.0 + c * f[i + 1]
        psi[i + 1] = num / denom

    # Normalize
    norm = np.sqrt(np.trapezoid(psi ** 2, x))
    if norm > 1e-12:
        psi /= norm

    return x, psi

