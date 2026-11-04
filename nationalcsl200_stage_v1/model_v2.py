import math

import torch
from torch import nn

from config_v2 import DEFAULT_CONFIG, V2Config


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding stored as a non-trainable buffer."""

    def __init__(self, d_model, max_length):
        super().__init__()
        if d_model <= 0 or max_length <= 0:
            raise ValueError("d_model and max_length must be positive.")

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            encoding[:, 1::2] = torch.cos(
                position * div_term[: encoding[:, 1::2].shape[1]]
            )

        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, D], got shape {tuple(x.shape)}.")
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError("Input sequence is longer than positional encoding.")
        if x.shape[2] != self.encoding.shape[2]:
            raise ValueError("Input embedding dimension does not match encoding.")
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)


class SignTransformerV2(nn.Module):
    def __init__(
        self,
        num_classes,
        config: V2Config = DEFAULT_CONFIG,
    ):
        super().__init__()
        config.validate()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        self.config = config
        self.num_classes = int(num_classes)

        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.input_activation = nn.GELU()
        self.position_encoding = SinusoidalPositionalEncoding(
            d_model=config.d_model,
            max_length=config.sequence_length,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        self.output_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, self.num_classes)

    def forward(self, x):
        expected_shape = (
            self.config.sequence_length,
            self.config.input_dim,
        )
        if x.ndim != 3 or tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                "SignTransformerV2 expects [B, T, F] with "
                f"(T, F)={expected_shape}, got {tuple(x.shape)}."
            )

        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.input_activation(x)
        x = self.position_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.output_norm(x)
        return self.classifier(x)


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
