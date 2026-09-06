# Deep Dive: PyTorch Autograd Internals in PINNs

In conventional Deep Learning, automatic differentiation calculates:

$$\nabla_\theta \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \theta}$$

where $\theta$ represents the trainable weights and biases of the neural network.

In **Physics-Informed Neural Networks (PINNs)**, we also differentiate the network output $u$ with respect to the **spatial and temporal input coordinates** $\mathbf{x} = (x, y, z, t)$:

$$\frac{\partial u}{\partial t}, \quad \nabla u = \left( \frac{\partial u}{\partial x}, \frac{\partial u}{\partial y} \right), \quad \Delta u = \sum_i \frac{\partial^2 u}{\partial x_i^2}$$

This document explains the underlying PyTorch computational mechanics, the role of `create_graph=True`, and performance considerations.

---

## 1. The Computational Graph for Coordinate Derivatives

To allow PyTorch to track operations on input coordinates, the coordinate tensor must explicitly request gradients:

```python
x = torch.linspace(0.0, 1.0, 100, requires_grad=True).unsqueeze(-1)
u = model(x)
```

When evaluating a differential operator like $\frac{du}{dx}$:

```python
du_dx = torch.autograd.grad(
    outputs=u,
    inputs=x,
    grad_outputs=torch.ones_like(u),
    create_graph=True,
    retain_graph=True,
)[0]
```

### Why `create_graph=True` is Mandatory

1. When `create_graph=False` (default), `torch.autograd.grad` evaluates the numerical vector-Jacobian product and discards the derivative's execution graph.
2. When `create_graph=True`, PyTorch **constructs a new subgraph** whose nodes represent the derivative operations.
3. Because the physics residual loss is a function of `du_dx`:
   $$\mathcal{L}_{\text{pde}} = \| \text{Residual}(u, \frac{\partial u}{\partial x}, \dots) \|^2$$
   Calling `loss.backward()` propagates gradients backwards through the physics operators, through the first derivative subgraph, and into the original network parameters $\theta$.
4. Without `create_graph=True`, calling `loss.backward()` raises an error: `element 0 of tensors does not require grad and does not have a grad_fn`.

---

## 2. Higher-Order Derivatives & The Laplacian

For 2nd-order PDEs (such as the Diffusion equation or Schrödinger equation), we need $\frac{\partial^2 u}{\partial x^2}$.

To compute $\frac{\partial^2 u}{\partial x^2}$ from $\frac{\partial u}{\partial x}$:
```python
# First derivative
du_dx = torch.autograd.grad(
    outputs=u,
    inputs=x,
    grad_outputs=torch.ones_like(u),
    create_graph=True,
    retain_graph=True,
)[0]

# Second derivative
d2u_dx2 = torch.autograd.grad(
    outputs=du_dx,
    inputs=x,
    grad_outputs=torch.ones_like(du_dx),
    create_graph=True,   # Keeps graph active for loss.backward()
    retain_graph=True,
)[0]
```

### Computational Graph Complexity
Each higher derivative expands the active DAG. For an $L$-layer MLP with activation $\sigma$:
- 0th order forward pass: $O(L \cdot W^2)$ FLOPs.
- 1st order derivative $\nabla_\mathbf{x} u$: $O(L \cdot W^2)$ backward FLOPs.
- 2nd order derivative $\Delta_\mathbf{x} u$: builds derivative of activation $\sigma''(z)$, requiring approximately $2\times$ to $3\times$ additional memory buffers.
- This is why smooth $C^\infty$ activations like $\tanh(z)$, $\text{SiLU}(z)$, or $\sin(z)$ are strictly required: piecewise-linear activations like ReLU have zero second derivative everywhere ($\text{ReLU}''(z) = 0$ for $z \ne 0$), rendering physics loss training completely impossible.

