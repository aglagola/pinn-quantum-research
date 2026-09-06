# Deep Dive: Gradient Pathology & Adaptive Loss Balancing

Training Physics-Informed Neural Networks is fundamentally a **multi-objective optimization problem**:

$$\min_\theta \mathcal{L}(\theta) = w_{\text{res}} \mathcal{L}_{\text{res}}(\theta) + w_{\text{bc}} \mathcal{L}_{\text{bc}}(\theta) + w_{\text{ic}} \mathcal{L}_{\text{ic}}(\theta) + w_{\text{norm}} \mathcal{L}_{\text{norm}}(\theta)$$

In standard tutorial implementations, practitioners set $w_{\text{res}} = w_{\text{bc}} = w_{\text{ic}} = 1.0$. This frequently fails to converge on realistic problems due to **gradient stiffness**.

---

## 1. The Gradient Pathology Problem

As analyzed by Wang, Teng, & Perdikaris (2021):
- The gradients of boundary condition losses $\nabla_\theta \mathcal{L}_{\text{bc}}$ often have magnitudes $10^2$ to $10^4 \times$ larger than the gradients of the physics residual $\nabla_\theta \mathcal{L}_{\text{res}}$.
- Boundary loss gradients originate from simple value matching on the boundaries, whereas residual loss gradients involve higher-order differential operators and non-linear interactions across the interior domain.
- Consequently, gradient descent is entirely dominated by boundary terms. The network rapidly learns trivial solutions that satisfy boundary conditions while completely violating physical laws inside the domain.

---

## 2. Learning Rate Annealing (Wang et al., 2021)

To counteract gradient pathology, **Learning Rate Annealing** dynamically rescales the auxiliary loss weights at each training step:

1. Compute the maximum gradient magnitude of the primary residual loss:
   $$\overline{|\nabla_\theta \mathcal{L}_{\text{res}}|} = \max_\theta |\nabla_\theta \mathcal{L}_{\text{res}}|$$

2. Compute the mean gradient magnitude for each auxiliary loss component $k \in \{\text{bc}, \text{ic}, \text{norm}\}$:
   $$\overline{|\nabla_\theta \mathcal{L}_k|} = \frac{1}{|\theta|} \sum_i \left| \frac{\partial \mathcal{L}_k}{\partial \theta_i} \right|$$

3. Set the target weight ratio:
   $$\hat{\lambda}_k = \frac{\max_\theta |\nabla_\theta \mathcal{L}_{\text{res}}|}{\overline{|\nabla_\theta \mathcal{L}_k|}}$$

4. Update via an Exponential Moving Average (EMA) with smoothing factor $\alpha \in [0.1, 0.2]$:
   $$\lambda_k^{(t+1)} = (1 - \alpha)\lambda_k^{(t)} + \alpha \hat{\lambda}_k$$

This ensures that the gradient contributions from the boundary, initial condition, and residual losses remain on the same order of magnitude throughout training.

---

## 3. Self-Adaptive Weights (SAPINN)

In Self-Adaptive PINNs (McClenny & Braga-Neto, 2023), each collocation point $\mathbf{x}_i$ is assigned an individual weight $\lambda_i \ge 0$:

$$\mathcal{L}(\theta, \mathbf{\lambda}) = \frac{1}{N} \sum_{i=1}^N \lambda_i | \mathcal{R}(\mathbf{x}_i; \theta) |^2$$

- Network parameters $\theta$ minimize $\mathcal{L}$ (gradient descent).
- Point weights $\lambda$ maximize $\mathcal{L}$ (gradient ascent).
- Hard or stiff regions (such as boundary layers, shock fronts, or barrier tunneling zones) naturally accumulate high weights, dynamically guiding the network's capacity to where the physics violation is greatest.

