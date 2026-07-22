from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

import logging

from kpi.models.STB.base_module import BaseModule

logger = logging.getLogger(__name__)

class SentenceEncoder(BaseModule):
    """SBERT-based sentence encoder.

    Constructor:
        model_name: Sentence-Transformers checkpoint name.
        embedding_dim: Expected embedding dimension, 384 by default.
        max_length: Tokenization cap.
        normalize_embeddings: L2-normalize output embeddings.

    Input shape:
        - A batch of sentences: [num_sentences]
        - A batch of documents: [batch, num_sentences]

    Output shape:
        - Flat input: [num_sentences, embedding_dim]
        - Nested input: [batch, max_sentences, embedding_dim]
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        max_length: int = 256,
        normalize_embeddings: bool = True,
        cache_folder: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.cache_folder = cache_folder
        # Log the initialization parameters for debugging purposes
        logger.info(
            "Initializing SentenceEncoder(model_name=%s, embedding_dim=%d, max_length=%d, normalize_embeddings=%s, cache_folder=%s)",
            model_name,
            embedding_dim,
            max_length,
            normalize_embeddings,
            cache_folder,
        )
        super().__init__(
            {
                "model_name": model_name,
                "embedding_dim": embedding_dim,
                "max_length": max_length,
                "normalize_embeddings": normalize_embeddings,
                "cache_folder": cache_folder,
            }
        )
        self.model = SentenceTransformer(model_name, cache_folder=cache_folder)
        self.model.max_seq_length = max_length
        logger.debug("Set SentenceEncoder model.max_seq_length=%d", max_length)
        if self.model.get_sentence_embedding_dimension() != embedding_dim:
            logger.warning(
                "Model embedding dimension %d does not match expected embedding_dim=%d",
                self.model.get_sentence_embedding_dimension(),
                embedding_dim,
            )

    def get_config(self) -> dict[str, Any]:
        return dict(self.config)

    def _encode_flat(self, sentences: Sequence[str]) -> Tensor:
        logger.debug("Encoding flat sentence batch of size=%d", len(sentences))
        embeddings = self.model.encode(
            list(sentences),
            batch_size=32,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        logger.debug("Encoded embeddings shape=%s", tuple(embeddings.shape))
        return embeddings

    def forward(
        self,
        sentences: Sequence[str] | Sequence[Sequence[str]],
        *,
        return_lengths: bool = False,
    ) -> Tensor | tuple[Tensor, list[int]]:
        """Encode either a flat sentence batch or a batch of documents."""

        if not sentences:
            logger.warning("SentenceEncoder.forward received empty sentences")
            raise ValueError("sentences must not be empty")

        first_item = sentences[0]
        if isinstance(first_item, str):
            if return_lengths:
                logger.warning("SentenceEncoder.forward received return_lengths=True for flat input")
                raise ValueError("return_lengths is only supported for nested document input")
            return self._encode_flat(sentences)  # type: ignore[arg-type]

        document_embeddings: list[Tensor] = []
        lengths: list[int] = []
        for document in sentences:  # type: ignore[assignment]
            doc_list = list(document)
            if not doc_list:
                logger.warning("SentenceEncoder.forward received an empty document")
                raise ValueError("documents must not be empty")
            embeddings = self._encode_flat(doc_list)
            document_embeddings.append(embeddings)
            lengths.append(len(doc_list))

        padded = pad_sequence(document_embeddings, batch_first=True)
        logger.debug("SentenceEncoder.forward padded output shape=%s", tuple(padded.shape))
        if return_lengths:
            return padded, lengths
        return padded

    def encode_documents(self, documents: Sequence[Sequence[str]]) -> Tensor:
        """Encode a batch of documents and return a padded tensor."""

        return self.forward(documents)

