"""Week 2 Harder Problem: Quantum Eigenvalue Discovery for TISE.

Solves:
    (-hbar^2 / 2m * d^2/dx^2 + V(x)) psi(x) = E * psi(x)

Key Innovations:
- Recovers both continuous wavefunctions psi(x) AND discrete quantized energy eigenvalues E.
- Normalization loss prevents the trivial solution psi(x) = 0.
- Gram-Schmidt orthogonality loss enables discovering excited states sequentially.
- Demonstrates on both the Quantum Harmonic Oscillator and the Double-Well tunneling potential.
"""

import os
import sys
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physics.quantum_schrodinger import (
    HarmonicOscillatorPotential,
    DoubleWellPotential,
    Schrodinger1D,
)
from src.solvers.quantum_eigen_solver import QuantumEigenSolver
from src.baselines.analytical import (
    qho_analytical_energy,
    qho_analytical_wavefunction,
)
from src.baselines.numerov_fd import solve_tise_finite_difference
from src.utils.metrics import (
    absolute_energy_error,
    relative_energy_error,
    wavefunction_fidelity,
)
from src.utils.plotting import (
    plot_quantum_eigenstate,
    plot_double_well_tunneling,
    plot_loss_history,
)


def run_quantum_harmonic_oscillator(device):
    print("\n=======================================================")
    print("Part A: Quantum Harmonic Oscillator (Eigenvalue Discovery)")
    print("=======================================================")

    pot = HarmonicOscillatorPotential(omega=1.0)
    tise = Schrodinger1D(potential=pot, x_range=(-5.0, 5.0))
    solver = QuantumEigenSolver(tise=tise, device=device)

    # 1. Ground State (n=0)
    model_0, E_0, hist_0 = solver.solve_state(
        state_index=0,
        E_init=0.7,
        epochs_adam=2000,
        epochs_lbfgs=250,
        w_pde=1.0,
        w_norm=12.0,
        w_ortho=0.0,
    )

    # 2. First Excited State (n=1) with orthogonality constraint
    model_1, E_1, hist_1 = solver.solve_state(
        state_index=1,
        E_init=1.8,
        epochs_adam=2200,
        epochs_lbfgs=250,
        w_pde=1.0,
        w_norm=12.0,
        w_ortho=25.0,
    )

    # Evaluation against Analytical Benchmarks
    out_dir = "results/week2_qho"
    os.makedirs(out_dir, exist_ok=True)

    x_test = np.linspace(-5.0, 5.0, 600)
    x_test_torch = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1).to(device)

    V_eval = 0.5 * (x_test ** 2)

    # Ground State Evaluation
    with torch.no_grad():
        psi_0_pred = model_0(x_test_torch).cpu().numpy().flatten()
    # Ensure correct phase
    if psi_0_pred[len(psi_0_pred) // 2] < 0:
        psi_0_pred = -psi_0_pred
    # Normalize
    norm_0 = np.sqrt(np.trapezoid(psi_0_pred ** 2, x_test))
    if norm_0 > 0:
        psi_0_pred /= norm_0

    psi_0_exact = qho_analytical_wavefunction(x_test, n=0)
    E_0_exact = qho_analytical_energy(0)
    fidelity_0 = wavefunction_fidelity(psi_0_pred, psi_0_exact, x_test)

    # Excited State Evaluation
    with torch.no_grad():
        psi_1_pred = model_1(x_test_torch).cpu().numpy().flatten()
    if psi_1_pred[int(len(psi_1_pred) * 0.75)] < 0:
        psi_1_pred = -psi_1_pred
    norm_1 = np.sqrt(np.trapezoid(psi_1_pred ** 2, x_test))
    if norm_1 > 0:
        psi_1_pred /= norm_1

    psi_1_exact = qho_analytical_wavefunction(x_test, n=1)
    E_1_exact = qho_analytical_energy(1)
    fidelity_1 = wavefunction_fidelity(psi_1_pred, psi_1_exact, x_test)

    print("\n-------------------------------------------------------")
    print("Quantum Harmonic Oscillator Eigenvalue Benchmark:")
    print(f"Ground State (n=0):")
    print(f"  Exact Energy:     {E_0_exact:.6f}")
    print(f"  PINN Energy:      {E_0:.6f} (Error: {absolute_energy_error(E_0, E_0_exact):.6f})")
    print(f"  State Fidelity:   {fidelity_0:.6f}")
    print(f"First Excited State (n=1):")
    print(f"  Exact Energy:     {E_1_exact:.6f}")
    print(f"  PINN Energy:      {E_1:.6f} (Error: {absolute_energy_error(E_1, E_1_exact):.6f})")
    print(f"  State Fidelity:   {fidelity_1:.6f}")
    print("-------------------------------------------------------")

    plot_quantum_eigenstate(
        x=x_test,
        psi_pred=psi_0_pred,
        psi_exact=psi_0_exact,
        potential=V_eval,
        energy_pred=E_0,
        energy_exact=E_0_exact,
        state_label="QHO Ground State (n=0)",
        save_path=os.path.join(out_dir, "qho_ground_state.png"),
    )

    plot_quantum_eigenstate(
        x=x_test,
        psi_pred=psi_1_pred,
        psi_exact=psi_1_exact,
        potential=V_eval,
        energy_pred=E_1,
        energy_exact=E_1_exact,
        state_label="QHO 1st Excited State (n=1)",
        save_path=os.path.join(out_dir, "qho_excited_state.png"),
    )

    plot_loss_history(hist_0, save_path=os.path.join(out_dir, "qho_loss_history.png"))


def run_quantum_double_well(device):
    print("\n=======================================================")
    print("Part B: Double-Well Potential & Quantum Tunneling")
    print("=======================================================")

    pot = DoubleWellPotential(a=1.5, lam=0.5)
    tise = Schrodinger1D(potential=pot, x_range=(-3.5, 3.5))
    solver = QuantumEigenSolver(tise=tise, device=device)

    # Finite difference numerical ground truth for double well
    x_fd, evals_fd, evecs_fd = solve_tise_finite_difference(
        potential_fn=lambda x: 0.5 * ((x ** 2 - 1.5 ** 2) ** 2),
        x_range=(-3.5, 3.5),
        n_points=1200,
        num_states=2,
    )

    # Symmetric ground state
    model_0, E_0, _ = solver.solve_state(
        state_index=0,
        E_init=0.8,
        ansatz_type="gaussian",
        epochs_adam=2000,
        epochs_lbfgs=250,
        w_pde=1.0,
        w_norm=15.0,
        w_ortho=0.0,
    )

    # Antisymmetric first excited state
    model_1, E_1, _ = solver.solve_state(
        state_index=1,
        E_init=1.3,
        ansatz_type="gaussian",
        epochs_adam=2200,
        epochs_lbfgs=250,
        w_pde=1.0,
        w_norm=15.0,
        w_ortho=30.0,
    )

    out_dir = "results/week2_double_well"
    os.makedirs(out_dir, exist_ok=True)

    x_test = np.linspace(-3.5, 3.5, 600)
    x_test_torch = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1).to(device)
    V_eval = 0.5 * ((x_test ** 2 - 1.5 ** 2) ** 2)

    with torch.no_grad():
        psi_0 = model_0(x_test_torch).cpu().numpy().flatten()
        psi_1 = model_1(x_test_torch).cpu().numpy().flatten()

    norm_0 = np.sqrt(np.trapezoid(psi_0 ** 2, x_test))
    norm_1 = np.sqrt(np.trapezoid(psi_1 ** 2, x_test))
    if norm_0 > 0:
        psi_0 /= norm_0
    if norm_1 > 0:
        psi_1 /= norm_1

    print("\n-------------------------------------------------------")
    print("Double-Well Parity Splitting (Tunneling):")
    print(f"Ground State E_0:      {E_0:.5f} (FD Truth: {evals_fd[0]:.5f})")
    print(f"1st Excited State E_1: {E_1:.5f} (FD Truth: {evals_fd[1]:.5f})")
    print(f"Energy Splitting dE:   {abs(E_1 - E_0):.5f} (FD Truth: {evals_fd[1] - evals_fd[0]:.5f})")
    print("-------------------------------------------------------")

    plot_double_well_tunneling(
        x=x_test,
        psi_0=psi_0,
        psi_1=psi_1,
        potential=V_eval,
        E0=E_0,
        E1=E_1,
        save_path=os.path.join(out_dir, "double_well_tunneling.png"),
    )


def main():
    device = (
        torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Running Week 2 Quantum Eigenvalue Solver on device: {device}")
    run_quantum_harmonic_oscillator(device)
    run_quantum_double_well(device)


if __name__ == "__main__":
    main()

