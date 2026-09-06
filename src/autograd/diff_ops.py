"""Automatic differentiation operators for Physics-Informed Neural Networks (PINNs).

In standard neural network training, autograd calculates dLoss/dWeight to update
network parameters. In PINNs, we also require gradients of the network output u
with respect to the input coordinates x (space) and t (time):
    e.g., du/dt, du/dx, d^2u/dx^2, \nabla^2 u.

Key Mechanics:
1. Input coordinate tensors MUST have `requires_grad=True`.
2. `torch.autograd.grad(..., create_graph=True)` builds the derivative computation
   into the PyTorch Directed Acyclic Graph (DAG). This enables subsequent backward
   passes of the physics loss (e.g. MSE(du/dt - alpha*d2u/dx2)) to backpropagate
   through the differential operators and reach the network weights theta.
3. `retain_graph=True` preserves the graph across multiple differential operations
   if intermediate tensors are reused.
"""

from typing import Optional, Sequence
import torch


def gradient(
    u: torch.Tensor,
    x: torch.Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> torch.Tensor:
    """Compute the first-order gradient of scalar field u with respect to coordinates x.

    Args:
        u: Output tensor of shape (N, 1) or scalar representing the field value.
        x: Input coordinate tensor of shape (N, D) where D is the dimension
           (e.g., [x] for 1D or [x, t] or [x, y]). Must have requires_grad=True.
        create_graph: If True, graph of the derivative will be constructed,
                      allowing computation of higher-order derivatives or backprop
                      w.r.t. network weights.
        retain_graph: If True, the graph used to compute the grad will be preserved.

    Returns:
        Tensor of shape (N, D) containing du/dx_i for each coordinate dimension.
    """
    if not x.requires_grad:
        raise ValueError("Input coordinate tensor x must have requires_grad=True.")

    grad_outputs = torch.ones_like(u)
    grad = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True,
    )[0]
    return grad


def nth_derivative(
    u: torch.Tensor,
    x: torch.Tensor,
    n: int = 1,
    component: Optional[int] = None,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> torch.Tensor:
    """Compute the n-th derivative of u with respect to coordinate x.

    Args:
        u: Output tensor of shape (N, 1).
        x: Input coordinate tensor of shape (N, D).
        n: Order of differentiation (e.g. 1 for du/dx, 2 for d^2u/dx^2).
        component: Optional column index of x to differentiate against.
                   If None and x has D=1, defaults to 0.
        create_graph: Whether to retain graph for subsequent differentiation.
        retain_graph: Whether to retain graph for memory reuse.

    Returns:
        Tensor of shape (N, 1) representing d^n u / dx_component^n.
    """
    if n < 1:
        raise ValueError(f"Derivative order n must be >= 1, got {n}")

    if component is None:
        if x.shape[-1] == 1:
            component = 0
        else:
            raise ValueError(
                f"Multi-dimensional input x (dim={x.shape[-1]}) requires specifying 'component'."
            )

    curr = u
    for i in range(n):
        # We need create_graph=True for all intermediate steps
        is_last = (i == n - 1)
        step_create = create_graph if is_last else True
        step_retain = retain_graph if is_last else True

        g = gradient(curr, x, create_graph=step_create, retain_graph=step_retain)
        curr = g[..., component : component + 1]

    return curr


def laplacian(
    u: torch.Tensor,
    x: torch.Tensor,
    spatial_indices: Optional[Sequence[int]] = None,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> torch.Tensor:
    r"""Compute the spatial Laplacian \Delta u = \sum_i d^2u / dx_i^2.

    Args:
        u: Output tensor of shape (N, 1).
        x: Input coordinate tensor of shape (N, D).
        spatial_indices: Optional list of column indices corresponding to spatial
                         coordinates (e.g., if x contains [x, y, t], pass [0, 1]
                         to exclude time t). If None, all coordinates are included.
        create_graph: Whether to construct the computation graph for the 2nd derivative.
        retain_graph: Whether to retain graph.

    Returns:
        Tensor of shape (N, 1) representing \Delta u.
    """
    if spatial_indices is None:
        spatial_indices = list(range(x.shape[-1]))

    # First gradient du/dx
    du_dx = gradient(u, x, create_graph=True, retain_graph=True)

    lap = torch.zeros_like(u)
    for idx in spatial_indices:
        # Differentiate each partial derivative du/dx_i w.r.t. x again
        d2u_dxi2 = torch.autograd.grad(
            outputs=du_dx[..., idx : idx + 1],
            inputs=x,
            grad_outputs=torch.ones_like(du_dx[..., idx : idx + 1]),
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True,
        )[0][..., idx : idx + 1]
        lap = lap + d2u_dxi2

    return lap


def divergence(
    u_vec: torch.Tensor,
    x: torch.Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> torch.Tensor:
    r"""Compute the divergence of a vector field: \nabla \cdot \mathbf{u} = \sum_i du_i/dx_i.

    Args:
        u_vec: Vector field tensor of shape (N, D).
        x: Coordinate tensor of shape (N, D).

    Returns:
        Scalar divergence tensor of shape (N, 1).
    """
    dim = u_vec.shape[-1]
    div = torch.zeros(u_vec.shape[0], 1, device=u_vec.device, dtype=u_vec.dtype)

    for i in range(dim):
        du_i = gradient(
            u_vec[..., i : i + 1],
            x,
            create_graph=create_graph,
            retain_graph=retain_graph,
        )
        div = div + du_i[..., i : i + 1]

    return div


def directional_derivative(
    u: torch.Tensor,
    x: torch.Tensor,
    direction: torch.Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> torch.Tensor:
    r"""Compute directional derivative \nabla u \cdot \mathbf{v}.

    Args:
        u: Scalar field of shape (N, 1).
        x: Coordinates of shape (N, D).
        direction: Unit direction vectors of shape (N, D) or (1, D).

    Returns:
        Tensor of shape (N, 1) representing the directional derivative.
    """
    grad_u = gradient(u, x, create_graph=create_graph, retain_graph=retain_graph)
    return (grad_u * direction).sum(dim=-1, keepdim=True)


def hessian_diagonal(
    u: torch.Tensor,
    x: torch.Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> torch.Tensor:
    """Compute the pure second partial derivatives [d2u/dx1^2, ..., d2u/dxD^2].

    Args:
        u: Scalar field of shape (N, 1).
        x: Coordinates of shape (N, D).

    Returns:
        Tensor of shape (N, D) with the diagonal elements of the Hessian.
    """
    d = x.shape[-1]
    du_dx = gradient(u, x, create_graph=True, retain_graph=True)
    d2_diag = []

    for i in range(d):
        d2 = torch.autograd.grad(
            outputs=du_dx[..., i : i + 1],
            inputs=x,
            grad_outputs=torch.ones_like(du_dx[..., i : i + 1]),
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True,
        )[0][..., i : i + 1]
        d2_diag.append(d2)

    return torch.cat(d2_diag, dim=-1)
