"""Research Experiment: Fixed Loss Weights vs Learning Rate Annealing.

Demonstrates Wang et al. (2021) Learning Rate Annealing on the 1D Heat Equation,
comparing convergence speed and final relative L2 accuracy against naive fixed weighting.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.mlp import PINNMLP
from src.physics.classical_pde import HeatEquation1D
from src.loss_balancing.fixed import FixedLossWeights
from src.loss_balancing.lr_annealing import LearningRateAnnealing
from src.solvers.pinn_trainer import PINNTrainer
from src.baselines.analytical import heat_1d_analytical
from src.utils.metrics import relative_l2_error
from src.utils.plotting import set_plot_style


def train_run(balancer, name, data, pde, device, epochs=2000):
    torch.manual_seed(42)
    model = PINNMLP(in_dim=2, out_dim=1, hidden_dims=[64, 64], activation="tanh").to(device)

    def compute_losses(net: nn.Module):
        res = pde.compute_residual(net, data["xt_f"])
        loss_res = torch.mean(res ** 2)
        u0 = net(data["xt_0"])
        loss_ic = torch.mean((u0 - data["u_0"]) ** 2)
        ub = net(data["xt_b"])
        loss_bc = torch.mean((ub - data["u_b"]) ** 2)

        return {
            "residual": loss_res,
            "initial": loss_ic,
            "boundary": loss_bc,
        }

    trainer = PINNTrainer(model=model, loss_balancer=balancer, lr=1e-3, device=device)
    print(f"\n--- Training with {name} ---")
    history = trainer.train_adam(compute_losses, epochs=epochs, log_interval=400)

    # Evaluate
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
    err = relative_l2_error(u_pred, u_exact)

    return history, err


def main():
    device = (
        torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Comparing Loss Balancing Strategies on device: {device}")

    pde = HeatEquation1D(alpha=0.4, L=1.0, T=1.0, k=1)
    torch.manual_seed(42)
    data = pde.sample_data(n_collocation=2000, n_initial=200, n_boundary=200, device=device)

    # 1. Fixed Weights
    fixed_balancer = FixedLossWeights({"residual": 1.0, "initial": 10.0, "boundary": 10.0})
    hist_fixed, err_fixed = train_run(fixed_balancer, "Fixed Weights (1, 10, 10)", data, pde, device)

    # 2. Learning Rate Annealing
    annealing_balancer = LearningRateAnnealing(
        aux_loss_keys=["initial", "boundary"],
        primary_key="residual",
        alpha=0.1,
        initial_weights={"initial": 1.0, "boundary": 1.0},
        update_freq=20,
    )
    hist_anneal, err_anneal = train_run(annealing_balancer, "LR Annealing (Wang et al.)", data, pde, device)

    print("\n=======================================================")
    print("Comparative Study Results:")
    print(f"Fixed Weights L2 Error:        {err_fixed:.6e} ({err_fixed * 100:.4f}%)")
    print(f"LR Annealing L2 Error:         {err_anneal:.6e} ({err_anneal * 100:.4f}%)")
    improvement = (err_fixed - err_anneal) / err_fixed * 100
    print(f"Accuracy Improvement:         {improvement:+.2f}%")
    print("=======================================================")

    # Plot comparison
    set_plot_style()
    out_dir = "results/loss_comparison"
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist_fixed["total_loss"], label=f"Fixed Weights (L2 Err: {err_fixed:.4f})", color="orange", alpha=0.8)
    ax.plot(hist_anneal["total_loss"], label=f"LR Annealing (L2 Err: {err_anneal:.4f})", color="teal", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Loss")
    ax.set_title("Loss Balancing Comparison: Fixed vs Learning Rate Annealing")
    ax.legend()

    plot_path = os.path.join(out_dir, "loss_weighting_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison figure to: {plot_path}")


if __name__ == "__main__":
    main()

