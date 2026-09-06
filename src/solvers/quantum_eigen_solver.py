"""Quantum Eigenvalue Solver for the Time-Independent Schrödinger Equation."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm
from src.models.quantum_net import QuantumPINN
from src.physics.quantum_schrodinger import Schrodinger1D


class QuantumEigenSolver:
    """Solver for discovering quantized quantum eigenstates and energy eigenvalues."""

    def __init__(
        self,
        tise: Schrodinger1D,
        device: Optional[torch.device] = None,
    ):
        self.tise = tise
        self.device = device or (
            torch.device("mps")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.discovered_states: List[QuantumPINN] = []
        self.discovered_energies: List[float] = []

    def solve_state(
        self,
        state_index: int = 0,
        E_init: float = 0.5,
        hidden_dims: List[int] = [64, 64, 64],
        activation: str = "tanh",
        ansatz_type: str = "gaussian",
        epochs_adam: int = 2000,
        epochs_lbfgs: int = 300,
        lr_adam: float = 1e-3,
        w_pde: float = 1.0,
        w_norm: float = 10.0,
        w_ortho: float = 20.0,
        n_colloc: int = 1000,
        n_dense: int = 500,
    ) -> Tuple[QuantumPINN, float, Dict[str, List[float]]]:
        """Train a QuantumPINN to find the n-th eigenstate and its energy eigenvalue."""
        print(f"\n--- Solving Quantum Eigenstate n={state_index} (Init E={E_init:.2f}) ---")

        model = QuantumPINN(
            in_dim=1,
            hidden_dims=hidden_dims,
            activation=activation,
            ansatz_type=ansatz_type,
            domain_bounds=(self.tise.x_min, self.tise.x_max),
            E_init=E_init,
        ).to(self.device)

        # Optimize network weights AND learnable scalar E
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam)

        # Sample domain points
        x_colloc, x_dense = self.tise.sample_domain(
            n_collocation=n_colloc,
            n_dense=n_dense,
            device=self.device,
        )

        history: Dict[str, List[float]] = {
            "total_loss": [],
            "pde_loss": [],
            "norm_loss": [],
            "ortho_loss": [],
            "energy": [],
        }

        # Stage 1: Adam optimization
        pbar = tqdm(range(epochs_adam), desc=f"State n={state_index} [Adam]")
        for epoch in pbar:
            optimizer.zero_grad()

            # 1. Physics PDE residual loss
            res = self.tise.compute_residual(model, x_colloc)
            loss_pde = torch.mean(res ** 2)

            # 2. Wavefunction normalization loss \int |\psi|^2 dx = 1
            loss_norm = self.tise.compute_normalization_loss(model, x_dense)

            # 3. Orthogonality against previous eigenstates
            loss_ortho = self.tise.compute_orthogonality_loss(
                model,
                self.discovered_states,
                x_dense,
            )

            total_loss = w_pde * loss_pde + w_norm * loss_norm + w_ortho * loss_ortho

            total_loss.backward()
            optimizer.step()

            # Log
            history["total_loss"].append(total_loss.item())
            history["pde_loss"].append(loss_pde.item())
            history["norm_loss"].append(loss_norm.item())
            history["ortho_loss"].append(loss_ortho.item())
            history["energy"].append(model.get_energy())

            if epoch % 200 == 0 or epoch == epochs_adam - 1:
                pbar.set_postfix({
                    "E": f"{model.get_energy():.4f}",
                    "PDE": f"{loss_pde.item():.2e}",
                    "Norm": f"{loss_norm.item():.2e}",
                })

        # Stage 2: L-BFGS fine-tuning (if requested)
        if epochs_lbfgs > 0:
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=0.5,
                max_iter=epochs_lbfgs,
                history_size=40,
                line_search_fn="strong_wolfe",
            )
            lbfgs_pbar = tqdm(total=epochs_lbfgs, desc=f"State n={state_index} [L-BFGS]")

            def closure():
                lbfgs.zero_grad()
                res = self.tise.compute_residual(model, x_colloc)
                loss_pde = torch.mean(res ** 2)
                loss_norm = self.tise.compute_normalization_loss(model, x_dense)
                loss_ortho = self.tise.compute_orthogonality_loss(
                    model,
                    self.discovered_states,
                    x_dense,
                )
                total = w_pde * loss_pde + w_norm * loss_norm + w_ortho * loss_ortho
                total.backward()

                lbfgs_pbar.update(1)
                lbfgs_pbar.set_postfix({"E": f"{model.get_energy():.4f}", "Loss": f"{total.item():.2e}"})
                history["energy"].append(model.get_energy())
                history["total_loss"].append(total.item())
                return total

            lbfgs.step(closure)
            lbfgs_pbar.close()

        final_E = model.get_energy()
        self.discovered_states.append(model)
        self.discovered_energies.append(final_E)
        print(f"-> Discovered State n={state_index}: Final E = {final_E:.5f}")

        return model, final_E, history

