from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch import Tensor, nn

from kpi.models.STB.base_module import BaseModule, lengths_to_padding_mask


logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Input shape: [batch, sequence_length, d_model]
    Output shape: same as input
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class GlobalTransformer(BaseModule):
    """Global self-attention encoder for lecture sentence features.

    Constructor:
        d_model: Feature width.
        n_heads: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        feedforward_dim: Feed-forward hidden size.
        dropout: Dropout probability.

    Input shape:
        [batch, sequence_length, d_model]

    Output shape:
        [batch, sequence_length, d_model]
    """

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        num_layers: int = 4,
        feedforward_dim: int = 1536,
        dropout: float = 0.1,
    ) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout
        super().__init__(
            {
                "d_model": d_model,
                "n_heads": n_heads,
                "num_layers": num_layers,
                "feedforward_dim": feedforward_dim,
                "dropout": dropout,
            }
        )
        logger.info(
            "Initializing GlobalTransformer(d_model=%d, n_heads=%d, num_layers=%d, feedforward_dim=%d, dropout=%.3f)",
            d_model,
            n_heads,
            num_layers,
            feedforward_dim,
            dropout,
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.position = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def get_config(self) -> dict[str, Any]:
        return dict(self.config)

    def forward(self, x: Tensor, lengths: Tensor | list[int] | None = None) -> Tensor:
        """Contextualize sentence features using global self-attention."""

        squeeze_batch = x.ndim == 2
        if squeeze_batch:
            x = x.unsqueeze(0)
        logger.debug("GlobalTransformer.forward input shape=%s, lengths_provided=%s", tuple(x.shape), lengths is not None)

        padding_mask = None
        if lengths is not None:
            padding_mask = lengths_to_padding_mask(lengths, x.size(1))

        x = self.input_norm(x)
        x = self.position(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if squeeze_batch:
            x = x.squeeze(0)
        logger.debug("GlobalTransformer.forward output shape=%s", tuple(x.shape))
        return x

