"""Interactive Plotly visualization utilities for the PINN Web UI."""

from typing import Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_wavefunction_figure(
    x: np.ndarray,
    psi_pred: np.ndarray,
    psi_exact: Optional[np.ndarray],
    potential: np.ndarray,
    energy_pred: float,
    energy_exact: Optional[float] = None,
    title: str = "Quantum Wavefunction & Energy Eigenvalue",
) -> go.Figure:
    """Create interactive Plotly figure of wavefunction, density, and potential well."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Wavefunction ψ(x)", "Probability Density |ψ(x)|² & Potential V(x)"),
        specs=[[{"secondary_y": False}, {"secondary_y": True}]],
    )

    # Subplot 1: Wavefunction
    if psi_exact is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=psi_exact,
                mode="lines",
                name="Exact ψ(x)",
                line=dict(color="#4A5568", dash="dash", width=2),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=psi_pred,
            mode="lines",
            name="PINN ψ(x)",
            line=dict(color="#E53E3E", width=2.5),
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Density |psi|^2
    fig.add_trace(
        go.Scatter(
            x=x,
            y=psi_pred ** 2,
            mode="lines",
            name="PINN |ψ|²",
            fill="tozeroy",
            fillcolor="rgba(229, 62, 62, 0.2)",
            line=dict(color="#E53E3E", width=2),
        ),
        row=1,
        col=2,
        secondary_y=False,
    )

    if psi_exact is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=psi_exact ** 2,
                mode="lines",
                name="Exact |ψ|²",
                line=dict(color="#4A5568", dash="dash", width=1.5),
            ),
            row=1,
            col=2,
            secondary_y=False,
        )

    # Secondary Y: Potential V(x)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=potential,
            mode="lines",
            name="Potential V(x)",
            line=dict(color="#3182CE", dash="dot", width=1.8),
        ),
        row=1,
        col=2,
        secondary_y=True,
    )

    # Energy horizontal lines
    fig.add_hline(
        y=energy_pred,
        line=dict(color="#E53E3E", dash="dashdot", width=1.5),
        annotation_text=f"E_PINN = {energy_pred:.4f}",
        annotation_position="top left",
        row=1,
        col=2,
        secondary_y=True,
    )

    if energy_exact is not None:
        fig.add_hline(
            y=energy_exact,
            line=dict(color="#2D3748", dash="dot", width=1.5),
            annotation_text=f"E_Exact = {energy_exact:.4f}",
            annotation_position="bottom left",
            row=1,
            col=2,
            secondary_y=True,
        )

    fig.update_layout(
        title_text=title,
        template="plotly_white",
        hovermode="x unified",
        height=480,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )

    fig.update_xaxes(title_text="Coordinate x", row=1, col=1)
    fig.update_xaxes(title_text="Coordinate x", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude ψ(x)", row=1, col=1)
    fig.update_yaxes(title_text="Density |ψ(x)|²", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Potential V(x) / Energy", row=1, col=2, secondary_y=True)

    return fig


def create_double_well_figure(
    x: np.ndarray,
    psi_0: np.ndarray,
    psi_1: np.ndarray,
    potential: np.ndarray,
    E0: float,
    E1: float,
) -> go.Figure:
    """Create interactive plot for double well parity splitting."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=x,
            y=psi_0,
            mode="lines",
            name=f"Ground State ψ₀ (Even, E₀={E0:.4f})",
            line=dict(color="#E53E3E", width=2.5),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=psi_1,
            mode="lines",
            name=f"1st Excited State ψ₁ (Odd, E₁={E1:.4f})",
            line=dict(color="#3182CE", width=2.5, dash="dash"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=potential,
            mode="lines",
            name="Double Well V(x)",
            line=dict(color="#718096", width=1.5, dash="dot"),
        ),
        secondary_y=True,
    )

    dE = abs(E1 - E0)
    fig.update_layout(
        title=f"Quantum Tunneling: Double-Well Parity Splitting (ΔE = {dE:.5f})",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )
    fig.update_xaxes(title_text="Coordinate x")
    fig.update_yaxes(title_text="Wavefunction Amplitude ψ(x)", secondary_y=False)
    fig.update_yaxes(title_text="Potential V(x)", secondary_y=True)

    return fig


def create_uq_figure(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    exact: Optional[np.ndarray] = None,
    collocation_x: Optional[np.ndarray] = None,
) -> go.Figure:
    """Create interactive plot for Deep Ensemble uncertainty quantification."""
    fig = go.Figure()

    # Shaded +/- 2 sigma band
    upper_bound = mean + 2 * std
    lower_bound = mean - 2 * std

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill="toself",
            fillcolor="rgba(49, 130, 206, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Epistemic Uncertainty (±2σ)",
        )
    )

    # Mean prediction
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            name="Ensemble Mean μ(x)",
            line=dict(color="#2B6CB0", width=2.5),
        )
    )

    # Exact solution
    if exact is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=exact,
                mode="lines",
                name="Exact Solution",
                line=dict(color="#2D3748", dash="dash", width=2),
            )
        )

    # Collocation training points
    if collocation_x is not None:
        fig.add_trace(
            go.Scatter(
                x=collocation_x,
                y=np.zeros_like(collocation_x),
                mode="markers",
                name="Collocation Points",
                marker=dict(color="#4A5568", size=5, opacity=0.4),
            )
        )

    fig.update_layout(
        title="Deep Ensemble Uncertainty Quantification (UQ)",
        template="plotly_white",
        hovermode="x unified",
        height=480,
        xaxis_title="Coordinate x",
        yaxis_title="Wavefunction ψ(x)",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )

    return fig


def create_2d_quantum_figure(
    x: np.ndarray,
    y: np.ndarray,
    density_2d: np.ndarray,
    energy: float,
) -> go.Figure:
    """Create 3D interactive surface plot of 2D quantum ground state."""
    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=density_2d,
                colorscale="Inferno",
                colorbar=dict(title="|ψ|²"),
            )
        ]
    )

    fig.update_layout(
        title=f"2D Quantum Ground State Probability Density |ψ(x, y)|² (E = {energy:.4f})",
        template="plotly_white",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="Probability Density",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        height=560,
    )

    return fig


def create_heat_equation_figure(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    u_pred: np.ndarray,
    u_exact: np.ndarray,
) -> go.Figure:
    """Create interactive heatmap / contour comparing PINN vs analytical heat equation."""
    error = np.abs(u_pred - u_exact)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("PINN Prediction u(x, t)", "Exact Analytical u(x, t)", "Pointwise Error |u - u_exact|"),
    )

    fig.add_trace(
        go.Heatmap(x=t_grid, y=x_grid, z=u_pred.T, colorscale="Viridis", colorbar=dict(x=0.28)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(x=t_grid, y=x_grid, z=u_exact.T, colorscale="Viridis", colorbar=dict(x=0.62)),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(x=t_grid, y=x_grid, z=error.T, colorscale="Magma", colorbar=dict(x=1.0)),
        row=1,
        col=3,
    )

    fig.update_layout(
        title="1D Diffusion: Heat Equation PINN Solution & Pointwise Residuals",
        template="plotly_white",
        height=450,
    )
    fig.update_xaxes(title_text="Time t", row=1, col=1)
    fig.update_xaxes(title_text="Time t", row=1, col=2)
    fig.update_xaxes(title_text="Time t", row=1, col=3)
    fig.update_yaxes(title_text="Space x", row=1, col=1)

    return fig
