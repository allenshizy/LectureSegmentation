from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor, nn

from kpi.models.STB.base_module import BaseModule, lengths_to_padding_mask


logger = logging.getLogger(__name__)


class BoundaryDetector(BaseModule):
    """BiLSTM boundary detector.

    Constructor:
        input_dim: Feature width entering the detector.
        hidden_dim: Hidden size of each LSTM direction.
        num_layers: Number of LSTM layers.
        bidirectional: Whether to use a bidirectional LSTM.
        dropout: Dropout used in the recurrent stack and classifier head.

    Input shape:
        [batch, sequence_length, input_dim]

    Output shape:
        [batch, sequence_length, 1]
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        super().__init__(
            {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "bidirectional": bidirectional,
                "dropout": dropout,
            }
        )
        logger.info(
            "Initializing BoundaryDetector(input_dim=%d, hidden_dim=%d, num_layers=%d, bidirectional=%s, dropout=%.3f)",
            input_dim,
            hidden_dim,
            num_layers,
            bidirectional,
            dropout,
        )
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        head_input_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def get_config(self) -> dict[str, Any]:
        return dict(self.config)

    def forward(
        self,
        x: Tensor,
        lengths: Tensor | list[int] | None = None,
    ) -> Tensor:
        """Predict boundary logits for each sentence position."""

        squeeze_batch = x.ndim == 2
        if squeeze_batch:
            x = x.unsqueeze(0)
        logger.debug("BoundaryDetector.forward input shape=%s, lengths_provided=%s", tuple(x.shape), lengths is not None)

        if lengths is not None and not torch.is_tensor(lengths):
            lengths = torch.as_tensor(lengths, dtype=torch.long)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.lstm(packed)
            sequence_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=x.size(1),
            )
        else:
            sequence_output, _ = self.lstm(x)

        logits = self.classifier(sequence_output)
        if lengths is not None:
            padding_mask = lengths_to_padding_mask(lengths, x.size(1))
            logits = logits.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if squeeze_batch:
            logits = logits.squeeze(0)
        logger.debug("BoundaryDetector.forward output shape=%s", tuple(logits.shape))
        return logits


class LinearBoundaryDetector(BaseModule):
    """Linear boundary detector used for probing representation quality.

    Input shape:
        [batch, sequence_length, input_dim]

    Output shape:
        [batch, sequence_length, 1]
    """

    def __init__(
        self,
        input_dim: int = 384,
        bias: bool = True,
    ) -> None:
        self.input_dim = input_dim
        self.bias = bias
        super().__init__(
            {
                "input_dim": input_dim,
                "bias": bias,
            }
        )
        logger.info(
            "Initializing LinearBoundaryDetector(input_dim=%d, bias=%s)",
            input_dim,
            bias,
        )
        self.classifier = nn.Linear(input_dim, 1, bias=bias)

    def get_config(self) -> dict[str, Any]:
        return dict(self.config)

    def forward(
        self,
        x: Tensor,
        lengths: Tensor | list[int] | None = None,
    ) -> Tensor:
        squeeze_batch = x.ndim == 2
        if squeeze_batch:
            x = x.unsqueeze(0)
        logger.debug(
            "LinearBoundaryDetector.forward input shape=%s, lengths_provided=%s",
            tuple(x.shape),
            lengths is not None,
        )

        if lengths is not None and not torch.is_tensor(lengths):
            lengths = torch.as_tensor(lengths, dtype=torch.long)

        logits = self.classifier(x)
        if lengths is not None:
            padding_mask = lengths_to_padding_mask(lengths, x.size(1))
            logits = logits.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if squeeze_batch:
            logits = logits.squeeze(0)
        logger.debug("LinearBoundaryDetector.forward output shape=%s", tuple(logits.shape))
        return logits

