from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor, nn

from kpi.models.STB.base_module import BaseModule, lengths_to_padding_mask


logger = logging.getLogger(__name__)


class BoundaryDetector(BaseModule):
    """Boundary detector that can run either as BiLSTM+MLP or pure MLP.

    Constructor:
        input_dim: Feature width entering the detector.
        hidden_dim: Hidden size of each LSTM direction when using the BiLSTM path.
        num_layers: Number of LSTM layers when using the BiLSTM path.
        bidirectional: Whether to use a bidirectional LSTM when using the BiLSTM path.
        dropout: Dropout used in the recurrent stack and classifier head.
        classifier_hidden_dim: Hidden size of the final classifier head. Use None for a linear head.
        classifier_dropout: Dropout in the final classifier head.
        use_mlp_only: When True, bypass the LSTM and run a pure MLP detector.

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
        classifier_hidden_dim: int | None = 128,
        classifier_dropout: float | None = None,
        use_mlp_only: bool = False,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.classifier_hidden_dim = classifier_hidden_dim
        self.classifier_dropout = dropout if classifier_dropout is None else classifier_dropout
        self.use_mlp_only = use_mlp_only
        super().__init__(
            {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "bidirectional": bidirectional,
                "dropout": dropout,
                "classifier_hidden_dim": classifier_hidden_dim,
                "classifier_dropout": self.classifier_dropout,
                "use_mlp_only": use_mlp_only,
            }
        )
        logger.info(
            "Initializing BoundaryDetector(input_dim=%d, hidden_dim=%d, num_layers=%d, bidirectional=%s, dropout=%.3f, classifier_hidden_dim=%s, classifier_dropout=%.3f, use_mlp_only=%s)",
            input_dim,
            hidden_dim,
            num_layers,
            bidirectional,
            dropout,
            classifier_hidden_dim,
            self.classifier_dropout,
            use_mlp_only,
        )
        if classifier_hidden_dim is None:
            classifier_head: nn.Module = nn.Linear(input_dim, 1)
        else:
            if classifier_hidden_dim <= 1:
                raise ValueError("classifier_hidden_dim must be > 1 when using an MLP head")
            classifier_head = nn.Sequential(
                nn.Linear(input_dim, classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.classifier_dropout),
                nn.Linear(classifier_hidden_dim, 1),
            )

        if use_mlp_only:
            self.lstm = None
            self.classifier = classifier_head
        else:
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            if classifier_hidden_dim is None:
                self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * (2 if bidirectional else 1), classifier_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.classifier_dropout),
                    nn.Linear(classifier_hidden_dim, 1),
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

        if self.use_mlp_only:
            sequence_output = x
        elif lengths is not None:
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
    """Linear or single-hidden-layer MLP boundary detector used for probing.

    Input shape:
        [batch, sequence_length, input_dim]

    Output shape:
        [batch, sequence_length, 1]
    """

    def __init__(
        self,
        input_dim: int = 384,
        bias: bool = True,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        classifier_hidden_dim: int | None = None,
        classifier_dropout: float | None = None,
    ) -> None:
        resolved_hidden_dim = classifier_hidden_dim if classifier_hidden_dim is not None else hidden_dim
        resolved_dropout = dropout if classifier_dropout is None else classifier_dropout
        self.input_dim = input_dim
        self.bias = bias
        self.hidden_dim = resolved_hidden_dim
        self.dropout = resolved_dropout
        super().__init__(
            {
                "input_dim": input_dim,
                "bias": bias,
                "hidden_dim": resolved_hidden_dim,
                "dropout": resolved_dropout,
                "classifier_hidden_dim": classifier_hidden_dim,
                "classifier_dropout": classifier_dropout,
            }
        )
        logger.info(
            "Initializing LinearBoundaryDetector(input_dim=%d, bias=%s, hidden_dim=%s, dropout=%.3f)",
            input_dim,
            bias,
            resolved_hidden_dim,
            resolved_dropout,
        )
        if resolved_hidden_dim is None:
            self.classifier = nn.Linear(input_dim, 1, bias=bias)
        else:
            if resolved_hidden_dim <= 0:
                raise ValueError("hidden_dim must be > 0 when provided")
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, resolved_hidden_dim, bias=bias),
                nn.GELU(),
                nn.Dropout(resolved_dropout),
                nn.Linear(resolved_hidden_dim, 1, bias=bias),
            )

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

