r"""Evaluation metrics for PINN approximations."""

from typing import Union
import numpy as np
import torch


def relative_l2_error(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
) -> float:
    r"""Compute relative L2 error: ||pred - target||_2 / ||target||_2."""
    if isinstance(pred, torch.Tensor):
        pred_arr = pred.detach().cpu().numpy().flatten()
    else:
        pred_arr = pred.flatten()

    if isinstance(target, torch.Tensor):
        target_arr = target.detach().cpu().numpy().flatten()
    else:
        target_arr = target.flatten()

    diff_norm = np.linalg.norm(pred_arr - target_arr)
    target_norm = np.linalg.norm(target_arr)

    if target_norm < 1e-12:
        return float(diff_norm)
    return float(diff_norm / target_norm)


def absolute_energy_error(pred_E: float, exact_E: float) -> float:
    r"""Compute absolute energy discrepancy: |E_pred - E_exact|."""
    return abs(pred_E - exact_E)


def relative_energy_error(pred_E: float, exact_E: float) -> float:
    r"""Compute relative energy percentage error."""
    if abs(exact_E) < 1e-12:
        return abs(pred_E - exact_E)
    return abs(pred_E - exact_E) / abs(exact_E)


def wavefunction_fidelity(
    psi_pred: np.ndarray,
    psi_exact: np.ndarray,
    x: np.ndarray,
) -> float:
    r"""Compute quantum state overlap / fidelity: |\langle \psi_{pred} | \psi_{exact} \rangle|^2."""
    overlap = np.trapezoid(psi_pred * psi_exact, x)
    return float(overlap ** 2)

