from abc import ABC, abstractmethod
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from ml_model.losses import LossMixin
from ml_model.modules.embedder import Embedder, MDNOutput, VectorEmbedder
from ml_model.sampler import SamplerMixin
from schemas.batch import Batch
from schemas.ink import DigitalInk
from train.tracker import Tracker


class LocalModel(nn.Module, LossMixin, SamplerMixin, ABC):
    def __init__(self):
        super().__init__()
        self.tracker: Tracker | None = None

    @abstractmethod
    def _losses(self, batch: Batch) -> dict[str, Tensor]: ...

    @abstractmethod
    def monitor(self, batch: Batch) -> None: ...

    def _forward(self, input: Tensor) -> Tensor | MDNOutput:
        """
        Forward pass through the model.

        Args:
            input: Input tensor of shape [batch_size, seq_len, hidden_dim]

        Returns:
            Either logits tensor [batch_size, seq_len, vocab_size] for token models,
            or MDNOutput tuple for vector models
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def local_model(self) -> Self:
        return self

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    # must call model(batch) (internally model.__call__) to activate DDP hooks
    def forward(self, batch: Batch) -> dict[str, Tensor]:
        return self._losses(batch)

    def init_weights(self):
        for module in self.modules():
            match module:
                case nn.Linear() | nn.Embedding():
                    torch.nn.init.normal_(module.weight, std=0.02)
                case nn.RMSNorm():
                    torch.nn.init.ones_(module.weight)
                case _:
                    pass

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_tracker(self, tracker: Tracker) -> None:
        self.tracker = tracker

    def _track_ink(self, ink: DigitalInk, task: str, caption: str | None = None) -> None:
        ink.visualise(name=f"{task}{f': {caption}' if caption else ''}")
        if self.tracker is not None:
            self.tracker.log_image(ink.to_image(), task, caption)

    def _prepare_batch(
        self,
        batch: Batch,
        context_embedder: Embedder | None,
        target_embedder: Embedder,
        context_attr: str | None,
        target_input_attr: str,
        target_target_attr: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Prepares batch tensors for training by concatenating context and target sequences.

        The context part comes first and has no loss computed (mask=False).
        The target part comes second and has loss computed (mask=True).

        Args:
            batch: Batch of instances
            context_embedder: Embedder for context part (e.g., char_embedder or repr_embedder).
                If None, no context is used.
            target_embedder: Embedder for target part (e.g., repr_embedder or char_embedder)
            context_attr: Attribute name for context (e.g., "char" or "repr").
                If None, no context is used.
            target_input_attr: Attribute name for target input (e.g., "repr_input" or "char_input")
            target_target_attr: Attribute name for target targets
                (e.g., "repr_target" or "char_target")

        Returns:
            Tuple of (padded_input, padded_target, padded_mask)
        """
        inputs, targets, masks = [], [], []

        for inst in batch.instances:
            # Embed target sequences
            target_input = target_embedder.embed(getattr(inst, target_input_attr))
            target_target = getattr(inst, target_target_attr)

            # Handle context if provided
            if context_embedder is not None and context_attr is not None:
                # Embed context
                context = context_embedder.embed(getattr(inst, context_attr))

                # Concatenate context + target for input
                inputs.append(torch.cat([context, target_input], dim=0))

                # Create dummy target for context (1D for tokens, 2D for vectors)
                if target_target.ndim == 1:
                    dummy = torch.zeros(context.shape[0], dtype=torch.long, device=self._device)
                else:
                    dummy = torch.zeros(
                        context.shape[0], target_target.shape[-1], device=self._device
                    )
                targets.append(torch.cat([dummy, target_target], dim=0))

                # Create mask (False for context, True for target)
                context_mask = torch.zeros(context.shape[0], dtype=torch.bool, device=self._device)
                target_mask = torch.ones(
                    target_input.shape[0], dtype=torch.bool, device=self._device
                )
                masks.append(torch.cat([context_mask, target_mask], dim=0))
            else:
                # No context, just use target
                inputs.append(target_input)
                targets.append(target_target)
                masks.append(
                    torch.ones(target_input.shape[0], dtype=torch.bool, device=self._device)
                )

        # Pad sequences
        input = pad_sequence(inputs, batch_first=True, padding_value=0)
        target = pad_sequence(targets, batch_first=True, padding_value=0)
        mask = pad_sequence(masks, batch_first=True, padding_value=0)

        return input, target, mask

    def _generate_sequence(
        self,
        context: Tensor | None,
        output_embedder: Embedder,
        bos: Tensor,
        eos: Tensor,
        max_len: int,
        temperature: float,
    ) -> Tensor:
        """
        Generic sequence generation method used by all local models.

        Args:
            instance: Instance containing metadata (is_token, repr_id, etc.)
            context: Optional context tensor [seq_len, hidden_dim] or None
            output_embedder: Embedder for the output sequence
            bos: Beginning-of-sequence token/vector
            eos: End-of-sequence token/vector
            max_len: Maximum generation length
            temperature: Sampling temperature

        Returns:
            Generated sequence tensor
        """
        gen = bos.unsqueeze(0)  # [1, 5] (vector) or [1] (token)
        for _ in range(max_len):
            gen_embed = output_embedder.embed(gen).unsqueeze(0)

            # Concatenate context with generated embeddings if context exists
            if context is not None:
                input = torch.cat([context.unsqueeze(0), gen_embed], dim=1)
            else:
                input = gen_embed

            # Forward pass through model-specific logic
            pred = self._forward(input)

            # Sample next token or vector based on embedder type
            if isinstance(output_embedder, VectorEmbedder):
                assert isinstance(pred, tuple)
                last_pred = tuple(tensor[0:1, -1] for tensor in pred)
                assert len(last_pred) == 5
                next_vector = self.sample_vector(last_pred, temperature=temperature)
                next_vector = next_vector.squeeze(0).squeeze(0)
                gen = torch.cat([gen, next_vector.unsqueeze(0)], dim=0)
                if torch.any(next_vector * eos == 1.0):
                    break
            else:
                assert isinstance(pred, Tensor)
                next_token = self.sample_token(pred[0, -1], temperature=temperature)
                gen = torch.cat([gen, next_token], dim=0)
                if int(next_token) == int(eos):
                    break

        return gen
