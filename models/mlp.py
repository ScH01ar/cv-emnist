from __future__ import annotations

from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        num_classes: int = 47,
        input_dim: int = 28 * 28,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers = [nn.Flatten()]
        in_features = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
