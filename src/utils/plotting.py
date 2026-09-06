r"""Publication-quality plotting and visualization routines."""

from typing import Dict, List, Optional
import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_plot_style():
    """Apply clean, academic styling to matplotlib plots."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_heat_comparison(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    u_pred: np.ndarray,
    u_exact: np.ndarray,
    save_path: str = "results/figures/heat_comparison.png",
):
    """Plot 3-panel heat equation comparison: Predicted, Analytical, and Error."""
    set_plot_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    X, T = np.meshgrid(x_grid, t_grid)
    error = np.abs(u_pred - u_exact)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    c1 = axes[0].pcolormesh(T, X, u_pred, cmap="viridis", shading="auto")
    axes[0].set_title("PINN Prediction $u(x, t)$")
    axes[0].set_xlabel("Time $t$")
    axes[0].set_ylabel("Space $x$")
    fig.colorbar(c1, ax=axes[0])

    c2 = axes[1].pcolormesh(T, X, u_exact, cmap="viridis", shading="auto")
    axes[1].set_title("Exact Analytical $u(x, t)$")
    axes[1].set_xlabel("Time $t$")
    fig.colorbar(c2, ax=axes[1])

    c3 = axes[2].pcolormesh(T, X, error, cmap="magma", shading="auto")
    axes[2].set_title("Pointwise Error $|u_{pred} - u_{exact}|$")
    axes[2].set_xlabel("Time $t$")
    fig.colorbar(c3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_quantum_eigenstate(
    x: np.ndarray,
    psi_pred: np.ndarray,
    psi_exact: np.ndarray,
    potential: np.ndarray,
    energy_pred: float,
    energy_exact: float,
    state_label: str = "Ground State (n=0)",
    save_path: str = "results/figures/qho_eigenstate.png",
):
    """Plot wavefunction, probability density, potential, and energy level."""
    set_plot_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Wavefunction comparison
    ax1.plot(x, psi_exact, "k--", label=f"Exact (E={energy_exact:.4f})")
    ax1.plot(x, psi_pred, "r-", label=f"PINN (E={energy_pred:.4f})")
    ax1.set_title(f"Wavefunction $\\psi(x)$ - {state_label}")
    ax1.set_xlabel("Coordinate $x$")
    ax1.set_ylabel("$\\psi(x)$")
    ax1.legend()

    # Probability density + Potential well overlay
    ax2.plot(x, psi_exact ** 2, "k--", label="Exact $|\\psi|^2$")
    ax2.plot(x, psi_pred ** 2, "r-", label="PINN $|\\psi|^2$")
    ax2.fill_between(x, psi_pred ** 2, alpha=0.2, color="red")

    ax_pot = ax2.twinx()
    ax_pot.plot(x, potential, "b:", alpha=0.6, label="Potential $V(x)$")
    ax_pot.axhline(energy_pred, color="red", linestyle="-.", alpha=0.6, label=f"$E_{{PINN}}={energy_pred:.3f}$")
    ax_pot.axhline(energy_exact, color="black", linestyle="--", alpha=0.6, label=f"$E_{{exact}}={energy_exact:.3f}$")
    ax_pot.set_ylabel("Energy / Potential $V(x)$", color="blue")
    ax_pot.set_ylim(0, max(np.max(potential) * 0.8, energy_exact * 2.5))

    ax2.set_title(f"Probability Density $|\\psi(x)|^2$ and Energy $E$")
    ax2.set_xlabel("Coordinate $x$")
    ax2.set_ylabel("Density $|\\psi(x)|^2$")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_double_well_tunneling(
    x: np.ndarray,
    psi_0: np.ndarray,
    psi_1: np.ndarray,
    potential: np.ndarray,
    E0: float,
    E1: float,
    save_path: str = "results/figures/double_well_tunneling.png",
):
    """Plot symmetric ground state and antisymmetric excited state showing tunnel splitting."""
    set_plot_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    ax1.plot(x, psi_0, "crimson", label=f"Symmetric $\\psi_0$ (Even, $E_0={E0:.4f}$)")
    ax1.plot(x, psi_1, "navy", linestyle="--", label=f"Antisymmetric $\\psi_1$ (Odd, $E_1={E1:.4f}$)")
    ax1.set_xlabel("Position $x$")
    ax1.set_ylabel("Wavefunction $\\psi(x)$")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, potential, "gray", alpha=0.4, linestyle=":", label="Double-Well $V(x)$")
    ax2.set_ylabel("Potential $V(x)$", color="gray")
    ax2.set_ylim(0, np.max(potential) * 1.05)

    delta_E = abs(E1 - E0)
    plt.title(f"Double-Well Quantum Tunneling: Parity Splitting $\\Delta E = {delta_E:.5f}$")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss_history(
    history: Dict[str, List[float]],
    save_path: str = "results/figures/loss_history.png",
):
    """Plot loss convergence curves on log scale."""
    set_plot_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(9, 5))
    for name, values in history.items():
        if values and len(values) > 0:
            plt.plot(values, label=name)

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Convergence History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_uq_uncertainty(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    exact: np.ndarray,
    collocation_x: Optional[np.ndarray] = None,
    save_path: str = "results/figures/uq_uncertainty_bands.png",
):
    """Plot ensemble mean with +/- 2 sigma confidence envelope."""
    set_plot_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(x, exact, "k--", label="Exact Solution")
    plt.plot(x, mean, "b-", label="Ensemble Mean $\\mu(x)$")
    plt.fill_between(
        x,
        mean - 2 * std,
        mean + 2 * std,
        color="royalblue",
        alpha=0.3,
        label=r"Epistemic Uncertainty ($\pm 2\sigma$)",
    )

    if collocation_x is not None:
        plt.scatter(
            collocation_x,
            np.zeros_like(collocation_x),
            color="black",
            s=8,
            alpha=0.3,
            label="Training Collocation Points",
        )

    plt.xlabel("Position $x$")
    plt.ylabel("Wavefunction $\\psi(x)$")
    plt.title("Physics-Informed Deep Ensemble Uncertainty Quantification")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
