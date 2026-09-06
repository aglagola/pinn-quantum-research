"""Multi-Layer Perceptron and Fourier Feature networks for PINNs."""

from typing import List, Optional, Union
import math
import torch
import torch.nn as nn


class SinActivation(nn.Module):
    """Sinusoidal activation function for SIREN architectures."""

    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class FourierFeatureEmbedding(nn.Module):
    r"""Random Fourier Features mapping: \gamma(x) = [\cos(2\pi B x), \sin(2\pi B x)].

    Mitigates spectral bias in coordinate-based neural networks (Tancik et al., 2020),
    allowing PINNs to fit higher-frequency spatial oscillations and steep wave fronts.
    """

    def __init__(self, in_features: int, num_features: int, scale: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.num_features = num_features
        # Gaussian random projection matrix B ~ N(0, scale^2)
        B = torch.randn(in_features, num_features) * scale
        self.register_buffer("B", B)

    @property
    def out_features(self) -> int:
        return 2 * self.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_features)
        # x_proj: (N, num_features)
        x_proj = 2.0 * math.pi * torch.matmul(x, self.B)
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class PINNMLP(nn.Module):
    r"""Configurable Multi-Layer Perceptron tailored for differential equation solving.

    Key PINN architectural features:
    1. Smooth C^\infty activations (Tanh, SiLU, Sine) to allow stable higher-order derivatives.
    2. Optional Fourier feature encoding for high-frequency modes.
    3. Proper Glorot/Xavier initialization suited to physical boundary decay.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        activation: str = "tanh",
        use_fourier_features: bool = False,
        fourier_dim: int = 32,
        fourier_scale: float = 1.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_fourier_features = use_fourier_features

        # Optional Fourier embedding
        if use_fourier_features:
            self.embedding = FourierFeatureEmbedding(
                in_features=in_dim,
                num_features=fourier_dim,
                scale=fourier_scale,
            )
            curr_dim = self.embedding.out_features
        else:
            self.embedding = None
            curr_dim = in_dim

        # Activation function selector
        act_lower = activation.lower()
        if act_lower == "tanh":
            act_fn = nn.Tanh
        elif act_lower in ("silu", "swish"):
            act_fn = nn.SiLU
        elif act_lower == "gelu":
            act_fn = nn.GELU
        elif act_lower == "sin":
            act_fn = SinActivation
        else:
            raise ValueError(f"Unsupported activation: {activation}. Choose tanh, silu, gelu, or sin.")

        layers: List[nn.Module] = []
        for h_dim in hidden_dims:
            linear = nn.Linear(curr_dim, h_dim)
            self._init_weights(linear, act_lower)
            layers.append(linear)
            layers.append(act_fn())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            curr_dim = h_dim

        # Final output layer
        final_layer = nn.Linear(curr_dim, out_dim)
        nn.init.xavier_normal_(final_layer.weight, gain=1.0)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        self.net = nn.Sequential(*layers)

    def _init_weights(self, layer: nn.Linear, act_name: str) -> None:
        if act_name == "tanh":
            nn.init.xavier_normal_(layer.weight, gain=5.0 / 3.0)
        elif act_name in ("silu", "gelu"):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        elif act_name == "sin":
            # SIREN uniform initialization
            fan_in = layer.weight.size(1)
            bound = math.sqrt(6.0 / fan_in)
            nn.init.uniform_(layer.weight, -bound, bound)
        else:
            nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            x = self.embedding(x)
        return self.net(x)
