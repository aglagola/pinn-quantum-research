r"""Closed-form analytical solutions for classical and quantum benchmarks."""

import math
import numpy as np
import scipy.special as sp


def heat_1d_analytical(
    x: np.ndarray,
    t: np.ndarray,
    alpha: float = 0.4,
    L: float = 1.0,
    k: int = 1,
) -> np.ndarray:
    r"""Exact analytical solution for 1D Heat Equation:

        u(x, t) = \sin(k \pi x / L) \exp(-\alpha (k \pi / L)^2 t)
    """
    spatial = np.sin((k * np.pi * x) / L)
    temporal = np.exp(-alpha * ((k * np.pi / L) ** 2) * t)
    return spatial * temporal


def qho_analytical_energy(n: int, omega: float = 1.0, hbar: float = 1.0) -> float:
    r"""Exact energy eigenvalue for Quantum Harmonic Oscillator:

        E_n = \hbar \omega (n + 1/2)
    """
    return hbar * omega * (n + 0.5)


def qho_analytical_wavefunction(
    x: np.ndarray,
    n: int = 0,
    omega: float = 1.0,
    m: float = 1.0,
    hbar: float = 1.0,
) -> np.ndarray:
    r"""Exact normalized wavefunction for Quantum Harmonic Oscillator eigenstate n:

        \psi_n(x) = \frac{1}{\sqrt{2^n n!}} \left(\frac{m\omega}{\pi\hbar}\right)^{1/4}
                    \exp\left(-\frac{m\omega x^2}{2\hbar}\right) H_n\left(\sqrt{\frac{m\omega}{\hbar}} x\right)
    """
    alpha_const = m * omega / hbar
    xi = np.sqrt(alpha_const) * x

    # Normalization constant
    norm_const = (
        (alpha_const / np.pi) ** 0.25
        / np.sqrt((2.0 ** n) * sp.factorial(n))
    )

    # Hermite polynomial H_n(xi)
    # sp.hermite returns a poly1d object
    H_n = sp.hermite(n)
    h_vals = H_n(xi)

    gaussian = np.exp(-0.5 * xi ** 2)
    return norm_const * gaussian * h_vals

