"""Week 1 Foundation: 1D Heat Equation PINN Solver and Validation.

Solves:
    u_t - alpha * u_xx = 0, x in [0, 1], t in [0, 1]
    u(0, t) = u(1, t) = 0
    u(x, 0) = sin(pi * x)

Demonstrates:
- Autograd differentiation w.r.t. input coordinates (x, t) via diff_ops.
- Multi-objective loss formulation (residual, initial condition, boundary condition).
- Two-stage optimization: Adam exploration + L-BFGS high-precision quadratic refinement.
- Exact analytical validation and relative L2 error benchmarking.
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.mlp import PINNMLP
from src.physics.classical_pde import HeatEquation1D
from src.loss_balancing.fixed import FixedLossWeights
from src.solvers.pinn_trainer import PINNTrainer
from src.baselines.analytical import heat_1d_analytical
from src.utils.metrics import relative_l2_error
from src.utils.plotting import plot_heat_comparison, plot_loss_history


def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "heat_1d.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = (
        torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"=== Week 1: 1D Heat Equation PINN ===")
    print(f"Using device: {device}")

    pde_cfg = cfg["pde"]
    net_cfg = cfg["network"]
    train_cfg = cfg["training"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # 1. Initialize PDE & Data Sampler
    pde = HeatEquation1D(
        alpha=pde_cfg["alpha"],
        L=pde_cfg["L"],
        T=pde_cfg["T"],
        k=pde_cfg["k"],
    )
    data = pde.sample_data(
        n_collocation=train_cfg["n_collocation"],
        n_initial=train_cfg["n_initial"],
        n_boundary=train_cfg["n_boundary"],
        device=device,
    )

    # 2. Build Neural Network
    model = PINNMLP(
        in_dim=net_cfg["in_dim"],
        out_dim=net_cfg["out_dim"],
        hidden_dims=net_cfg["hidden_dims"],
        activation=net_cfg["activation"],
        use_fourier_features=net_cfg["use_fourier_features"],
    ).to(device)

    # 3. Define Loss Function
    loss_weights = FixedLossWeights(train_cfg["weights"])

    def compute_losses(net: nn.Module):
        # Physics residual loss
        res = pde.compute_residual(net, data["xt_f"])
        loss_res = torch.mean(res ** 2)

        # Initial condition loss
        u0_pred = net(data["xt_0"])
        loss_ic = torch.mean((u0_pred - data["u_0"]) ** 2)

        # Boundary condition loss
        ub_pred = net(data["xt_b"])
        loss_bc = torch.mean((ub_pred - data["u_b"]) ** 2)

        return {
            "residual": loss_res,
            "initial": loss_ic,
            "boundary": loss_bc,
        }

    # 4. Train with Adam + L-BFGS
    trainer = PINNTrainer(model=model, loss_balancer=loss_weights, lr=train_cfg["lr_adam"], device=device)

    print("\nStarting Stage 1: Adam Optimizer...")
    trainer.train_adam(compute_losses, epochs=train_cfg["epochs_adam"], log_interval=250)

    if train_cfg.get("epochs_lbfgs", 0) > 0:
        print("\nStarting Stage 2: L-BFGS Quasi-Newton Optimizer...")
        trainer.train_lbfgs(compute_losses, max_iter=train_cfg["epochs_lbfgs"])

    # 5. Evaluate against Analytical Baseline
    print("\nEvaluating against analytical baseline...")
    nx, nt = 100, 100
    x_test = np.linspace(0, pde.L, nx)
    t_test = np.linspace(0, pde.T, nt)
    X, T = np.meshgrid(x_test, t_test)

    xt_eval = torch.tensor(
        np.stack([X.flatten(), T.flatten()], axis=-1),
        dtype=torch.float32,
    ).to(device)

    model.eval()
    with torch.no_grad():
        u_pred = model(xt_eval).cpu().numpy().reshape(nt, nx)

    u_exact = heat_1d_analytical(X, T, alpha=pde.alpha, L=pde.L, k=pde.k)

    l2_err = relative_l2_error(u_pred, u_exact)
    max_err = np.max(np.abs(u_pred - u_exact))

    print(f"\n==========================================")
    print(f"Results for 1D Heat Equation PINN:")
    print(f"Relative L2 Error: {l2_err:.6e} ({l2_err * 100:.4f}%)")
    print(f"Maximum Pointwise Absolute Error: {max_err:.6e}")
    print(f"==========================================")

    # 6. Save Plots
    fig_path = os.path.join(out_dir, "heat_comparison.png")
    plot_heat_comparison(x_test, t_test, u_pred, u_exact, save_path=fig_path)
    print(f"Saved comparison plot to: {fig_path}")

    loss_fig_path = os.path.join(out_dir, "heat_loss_history.png")
    plot_loss_history(trainer.history, save_path=loss_fig_path)
    print(f"Saved loss curves to: {loss_fig_path}")


if __name__ == "__main__":
    main()

