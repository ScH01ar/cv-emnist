from __future__ import annotations

import math

import torch
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


class TokenBatchNorm(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, tokens, embed_dim]
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)


def build_token_norm(name: str, embed_dim: int) -> nn.Module:
    key = name.lower()
    if key == "none":
        return nn.Identity()
    if key == "layernorm":
        return nn.LayerNorm(embed_dim)
    if key == "batchnorm":
        return TokenBatchNorm(embed_dim)
    raise ValueError(f"Unsupported normalization: {name}")


class MLPBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = build_activation(activation)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: str = "gelu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = build_token_norm(norm, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.drop_path1 = nn.Dropout(dropout)

        self.norm2 = build_token_norm(norm, embed_dim)
        self.mlp = MLPBlock(
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = residual + self.drop_path1(attn_out)

        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})."
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 47,
        image_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        activation: str = "gelu",
        norm: str = "layernorm",
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        classifier: str = "cls",
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.classifier = classifier.lower()
        if self.classifier not in {"cls", "mean"}:
            raise ValueError("classifier must be either 'cls' or 'mean'.")

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        token_count = num_patches + (1 if self.classifier == "cls" else 0)

        if self.classifier == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.register_parameter("cls_token", None)

        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        mlp_dim = int(embed_dim * mlp_ratio)
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

        self.final_norm = build_token_norm(norm, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                fan_out //= module.groups
                module.weight.data.normal_(0.0, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.encoder(x)
        x = self.final_norm(x)

        if self.classifier == "cls":
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        return self.head(x)