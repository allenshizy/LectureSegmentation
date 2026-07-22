"""Pre-training model for STB backbone (SentenceEncoder + GlobalTransformer).

Two self-supervised objectives are jointly optimised:

* **Sentence Order Prediction (SOP)** – A random contiguous block of sentence
  embeddings is permuted inside a document.  The model predicts, *per position*,
  whether that sentence was displaced from its natural order (binary BCE).

* **Masked Sentence Reconstruction (MSR)** – A fraction of sentence embeddings
  are replaced by a learnable ``[MASK]`` token.  The model reconstructs the
  original SBERT embedding at each masked position (MSE regression).

After pre-training, call ``model.encoder.save_checkpoint(path)`` and
``model.transformer.save_checkpoint(path)`` to export weights that can be
loaded directly into a :class:`~kpi.models.STB.LectureSegmentationModel`.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import torch
from torch import Tensor, nn

from kpi.models.STB.base_module import BaseModule, lengths_to_padding_mask
from kpi.models.STB.global_transformer import GlobalTransformer
from kpi.models.STB.sentence_encoder import SentenceEncoder


logger = logging.getLogger(__name__)


class STBPretrainingModel(BaseModule):
    """Pre-training wrapper for SentenceEncoder + GlobalTransformer.

    Constructor args:
        encoder: Optional prebuilt :class:`SentenceEncoder`.
        transformer: Optional prebuilt :class:`GlobalTransformer`.
        mask_rate: Fraction of valid sentence positions masked per document for
            MSR.  Default ``0.15``.
        sop_prob: Per-document probability that SOP block-shuffling is applied.
            Default ``0.5``.
        sop_shuffle_ratio: Fraction of total document length to shuffle for SOP.
            Default ``0.15`` (i.e. 15% of the sentences are permuted as a block).
        encoder_kwargs: Keyword args forwarded to :class:`SentenceEncoder` when
            no *encoder* is provided.
        transformer_kwargs: Keyword args forwarded to :class:`GlobalTransformer`
            when no *transformer* is provided.

    During ``model.train()``, the forward pass applies both SOP and MSR
    perturbations and returns task logits / targets.  During ``model.eval()``
    the perturbations are skipped so the model acts as a pure encoder.

    Gradient notes:
        ``SentenceEncoder`` uses ``sentence_transformers.SentenceTransformer``
        internally which runs under ``torch.no_grad()``, so SBERT weights are
        effectively frozen.  Pre-training therefore updates the
        ``GlobalTransformer``, the learnable ``mask_token``, and the two task
        heads (``sop_head``, ``msr_head``).
    """

    def __init__(
        self,
        encoder: SentenceEncoder | None = None,
        transformer: GlobalTransformer | None = None,
        *,
        mask_rate: float = 0.15,
        sop_prob: float = 0.5,
        sop_shuffle_ratio: float = 0.15,
        encoder_kwargs: dict[str, Any] | None = None,
        transformer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        encoder_kwargs = dict(encoder_kwargs or {})
        transformer_kwargs = dict(transformer_kwargs or {})

        if encoder is None:
            encoder = SentenceEncoder(**encoder_kwargs)
        if transformer is None:
            transformer = GlobalTransformer(**transformer_kwargs)

        d_model: int = transformer.d_model

        super().__init__(
            {
                "mask_rate": mask_rate,
                "sop_prob": sop_prob,
                "sop_shuffle_ratio": sop_shuffle_ratio,
                "encoder": encoder.get_config(),
                "transformer": transformer.get_config(),
            }
        )

        self.encoder = encoder
        self.transformer = transformer
        self.mask_rate = mask_rate
        self.sop_prob = sop_prob
        self.sop_shuffle_ratio = sop_shuffle_ratio

        # Learnable [MASK] embedding; initialised to zero and trained.
        self.mask_token: nn.Parameter = nn.Parameter(torch.zeros(d_model))

        # Per-sentence binary head for SOP (displaced / not displaced)
        self.sop_head = nn.Linear(d_model, 1)

        # Per-position projection head for MSR (reconstruct SBERT embedding)
        self.msr_head = nn.Linear(d_model, d_model)

        logger.info(
            "Initialised STBPretrainingModel(d_model=%d, mask_rate=%.2f, sop_prob=%.2f, "
            "sop_shuffle_ratio=%.2f)",
            d_model,
            mask_rate,
            sop_prob,
            sop_shuffle_ratio,
        )

    def get_config(self) -> dict[str, Any]:
        return dict(self.config)

    # ------------------------------------------------------------------
    # Internal perturbation helpers
    # ------------------------------------------------------------------

    def _apply_sop(
        self,
        embeddings: Tensor,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Randomly shuffle a contiguous sentence block proportional to doc length.

        Args:
            embeddings: ``[B, L, D]`` SBERT embeddings (detached).
            lengths: ``[B]`` number of valid sentences per document.

        Returns:
            shuffled: ``[B, L, D]`` — same as *embeddings* except for shuffled
                blocks.
            sop_labels: ``[B, L]`` float32 — 1 at positions that were
                displaced, 0 otherwise.
        """
        B, L, _D = embeddings.shape
        shuffled = embeddings.clone()
        sop_labels = torch.zeros(B, L, dtype=torch.float32, device=embeddings.device)

        for i in range(B):
            n = int(lengths[i].item())
            if n < 2 or random.random() > self.sop_prob:
                continue

            # Calculate block size as a proportion of total length (min 2 sentences)
            block_size = max(2, int(round(n * self.sop_shuffle_ratio)))
            block_size = min(block_size, n)

            start = random.randint(0, n - block_size)

            perm = torch.randperm(block_size, device=embeddings.device)
            shuffled[i, start : start + block_size] = embeddings[i, start : start + block_size][perm]

            for j, p in enumerate(perm.tolist()):
                if j != p:
                    sop_labels[i, start + j] = 1.0

        return shuffled, sop_labels

    def _apply_msr(
        self,
        shuffled_embs: Tensor,
        lengths: Tensor,
        original_embs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Replace a random fraction of sentence embeddings with mask_token.

        The MSR reconstruction target for each masked position is the
        original unperturbed embedding at that position (i.e. the natural
        semantic content that should appear there in original order).

        Args:
            shuffled_embs: ``[B, L, D]`` — SOP-perturbed embeddings (detached).
            lengths: ``[B]`` number of valid sentences per document.
            original_embs: ``[B, L, D]`` — original unperturbed SBERT embeddings.

        Returns:
            masked_embs: ``[B, L, D]`` — input with mask token applied.
                Participates in the autograd graph via ``self.mask_token``.
            msr_targets: ``[N_masked, D]`` — original (pre-mask, pre-shuffle)
                embeddings at masked positions; detached from the graph.
            msr_mask: ``[B, L]`` bool — True where masking was applied.
        """
        B, L, _D = shuffled_embs.shape
        msr_mask = torch.zeros(B, L, dtype=torch.bool, device=shuffled_embs.device)

        for i in range(B):
            n = int(lengths[i].item())
            if n == 0:
                continue
            n_mask = max(1, int(round(n * self.mask_rate)))
            idx = torch.randperm(n, device=shuffled_embs.device)[:n_mask]
            msr_mask[i, idx] = True

        # Save original unperturbed reconstruction targets before masking
        msr_targets: Tensor = original_embs[msr_mask].detach()

        # Replace masked positions with learnable mask_token (stays in graph)
        mask_expanded = msr_mask.unsqueeze(-1).expand_as(shuffled_embs)
        masked_embs = torch.where(
            mask_expanded,
            self.mask_token.expand_as(shuffled_embs),
            shuffled_embs,
        )

        return masked_embs, msr_targets, msr_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        text: list[list[str]],
        lengths: Tensor,
        apply_perturbations: bool | None = None,
    ) -> dict[str, Tensor]:
        """Encode, perturb, contextualise, and produce pre-training outputs.

        During ``model.train()`` SOP and MSR perturbations are applied.
        During ``model.eval()`` perturbations are skipped by default, but can
        be enabled by setting *apply_perturbations=True* for validation loss
        computation. When perturbations are disabled, the model acts as a
        plain encoder (useful for downstream feature extraction after
        pre-training).

        Args:
            text: Batch of documents, each a list of sentence strings.
            lengths: ``[B]`` int tensor — number of sentences per document.
            apply_perturbations: If ``True``, apply SOP and MSR perturbations
                during eval mode (for validation loss). If ``None`` (default),
                use perturbations in train mode only. If ``False``, never
                apply perturbations.

        Returns:
            A dict with keys:

            ``sop_logits``  ``[B, L]``       raw SOP scores (use BCEWithLogits).
            ``sop_labels``  ``[B, L]``       float32 binary ground-truth (1=displaced).
            ``msr_preds``   ``[N, D]``       MSR head output for masked positions.
            ``msr_targets`` ``[N, D]``       SBERT embedding targets (detached).
            ``msr_mask``    ``[B, L]`` bool  True at masked positions.
            ``valid_mask``  ``[B, L]`` bool  True at non-padded positions.
        """
        # ---- 1. SBERT encoding (no gradients through encoder) ----------
        embeddings: Tensor = self.encoder(text)   # [B, L, D]
        B, L, D = embeddings.shape
        device = embeddings.device

        # Non-padded position mask
        pad_mask = lengths_to_padding_mask(lengths, L)   # True = padding
        valid_mask = ~pad_mask                            # True = valid

        # Determine whether to apply perturbations (default: only in training)
        if apply_perturbations is None:
            apply_perturbations = self.training

        # ---- 2. SOP perturbation (train or explicit apply_perturbations) ---
        if apply_perturbations:
            shuffled_embs, sop_labels = self._apply_sop(
                embeddings.detach(), lengths
            )
        else:
            shuffled_embs = embeddings.detach()
            sop_labels = torch.zeros(B, L, dtype=torch.float32, device=device)

        # ---- 3. MSR perturbation (train or explicit apply_perturbations) ---
        if apply_perturbations:
            masked_embs, msr_targets, msr_mask = self._apply_msr(
                shuffled_embs, lengths, original_embs=embeddings.detach()
            )
        else:
            masked_embs = shuffled_embs
            msr_targets = torch.zeros(0, D, device=device)
            msr_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        # ---- 4. Global transformer -------------------------------------
        features: Tensor = self.transformer(masked_embs, lengths)   # [B, L, D]

        # ---- 5. Task heads ---------------------------------------------
        sop_logits: Tensor = self.sop_head(features).squeeze(-1)    # [B, L]
        if msr_mask.any():
            msr_preds: Tensor = self.msr_head(features[msr_mask])   # [N, D]
        else:
            msr_preds = torch.zeros(0, D, device=device)

        logger.debug(
            "STBPretrainingModel forward: B=%d L=%d sop_displaced=%.3f msr_masked=%d",
            B,
            L,
            sop_labels[valid_mask].mean().item() if valid_mask.any() else 0.0,
            int(msr_mask.sum().item()),
        )

        return {
            "sop_logits": sop_logits,
            "sop_labels": sop_labels,
            "msr_preds": msr_preds,
            "msr_targets": msr_targets,
            "msr_mask": msr_mask,
            "valid_mask": valid_mask,
        }
