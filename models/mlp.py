from __future__ import annotations

from torch import nn


def build_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    if key == "elu":
        return nn.ELU()
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def build_norm(name: str, hidden_dim: int) -> nn.Module | None:
    key = name.lower()
    if key == "none":
        return None
    if key == "batchnorm":
        return nn.BatchNorm1d(hidden_dim)
    if key == "layernorm":
        return nn.LayerNorm(hidden_dim)
    raise ValueError(f"Unsupported normalization: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        num_classes: int = 47,
        input_dim: int = 28 * 28,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        activation: str = "relu",
        norm: str = "none",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers = [nn.Flatten()]
        in_features = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            norm_layer = build_norm(norm, hidden_dim)
            if norm_layer is not None:
                layers.append(norm_layer)
            layers.append(build_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
