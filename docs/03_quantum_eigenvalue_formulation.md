# Deep Dive: Quantum Eigenvalue Problems via PINNs

The Time-Independent Schrödinger Equation (TISE) is an **eigenvalue problem**:

$$\hat{H} \psi(x) = E \psi(x)$$

where the Hamiltonian operator $\hat{H}$ in 1D is:

$$\hat{H} = -\frac{\hbar^2}{2m} \frac{d^2}{dx^2} + V(x)$$

Unlike boundary-value problems where all equation parameters are known, in an eigenvalue problem **both the eigenfunction $\psi(x)$ and the energy eigenvalue $E$ are unknown**.

---

## 1. Energy $E$ as a Trainable Parameter

In this framework, the neural network models the spatial envelope of the wavefunction $\psi(x; \theta)$, while the energy eigenvalue $E$ is declared as a dedicated trainable scalar:

```python
self.E = nn.Parameter(torch.tensor(E_init, dtype=torch.float32))
```

During backpropagation, both $\theta$ (the network weights) and $E$ receive gradients from the physics residual loss:

$$\mathcal{L}_{\text{pde}}(\theta, E) = \frac{1}{N} \sum_{i=1}^N \left| -\frac{\hbar^2}{2m} \psi''(x_i; \theta) + (V(x_i) - E)\psi(x_i; \theta) \right|^2$$

$$\frac{\partial \mathcal{L}_{\text{pde}}}{\partial E} = -\frac{2}{N} \sum_{i=1}^N \psi(x_i; \theta) \left( \hat{H}\psi(x_i; \theta) - E \psi(x_i; \theta) \right)$$

When the wavefunction shape $\psi$ approximates an eigenstate, gradient descent drives $E$ toward the corresponding expectation value $\langle \hat{H} \rangle$.

---

## 2. Preventing Trivial Solution Collapse ($\psi(x) \equiv 0$)

Because $\hat{H}$ is a linear differential operator, $\psi(x) \equiv 0$ is a mathematically valid solution to $\hat{H}\psi = E\psi$ for any $E$. A naive neural network will instantly collapse all weights to zero to minimize the loss to 0.

To eliminate this failure mode, we introduce the **Wavefunction Normalization Loss**:

$$\mathcal{L}_{\text{norm}} = \left( \int_\Omega |\psi(x; \theta)|^2 dx - 1 \right)^2$$

Evaluated using high-order trapezoidal quadrature over a dense coordinate grid:

```python
psi_sq = (model(x_dense).squeeze(-1)) ** 2
integral = torch.trapezoid(psi_sq, x_dense.squeeze(-1))
loss_norm = (integral - 1.0) ** 2
```

---

## 3. Discovering Excited States via Gram-Schmidt Orthogonality

In quantum mechanics, eigenstates belonging to different energy levels are strictly orthogonal:

$$\langle \psi_j | \psi_k \rangle = \int_\Omega \psi_j^*(x) \psi_k(x) dx = \delta_{jk}$$

To discover the $n$-th excited state without collapsing back to the ground state ($n=0$), we enforce **Gram-Schmidt Orthogonality Loss** against all previously discovered eigenstates $\{\psi_0, \dots, \psi_{n-1}\}$:

$$\mathcal{L}_{\text{ortho}} = \sum_{k=0}^{n-1} \left| \int_\Omega \psi_k(x) \psi_n(x; \theta) dx \right|^2$$

When optimizing state $n$, the parameters of prior states $\{\psi_0, \dots, \psi_{n-1}\}$ are frozen (`torch.no_grad()`). The optimizer is thereby constrained to explore only the orthogonal subspace, naturally locking onto the next quantized energy level.

