from abc import ABC, abstractmethod
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from ml_model.losses import LossMixin
from ml_model.modules.embedder import Embedder, MDNOutput, VectorEmbedder
from ml_model.sampler import SamplerMixin
from ml_trainer.tracker import Tracker
from schemas.batch import Batch
from schemas.ink import DigitalInk

type ForwardOutput = Tensor | MDNOutput
type KVCaches = list[tuple[Tensor, Tensor]]


class LocalModel(nn.Module, LossMixin, SamplerMixin, ABC):
    def __init__(self):
        super().__init__()
        self.tracker: Tracker | None = None

    @abstractmethod
    def _losses(self, batch: Batch) -> dict[str, Tensor]: ...

    @abstractmethod
    def monitor(self, batch: Batch) -> None: ...

    def _forward(
        self,
        input: Tensor,
        start_pos: int = 0,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
    ) -> ForwardOutput | tuple[ForwardOutput, KVCaches]:
        """
        Forward pass through the model with KV cache support.
        """
        raise NotImplementedError("Subclasses must implement this method for cached generation")

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

    def _generate_sequences(
        self,
        context: Tensor | None,
        output_embedder: Embedder,
        bos: Tensor,
        eos: Tensor,
        max_len: int,
        temperature: float,
        num_generations: int,
    ) -> list[Tensor]:
        """
        Vectorized sequence generation using KV cache for efficiency.

        Args:
            context: Optional context tensor [seq_len, hidden_dim] or None
            output_embedder: Embedder for the output sequence
            bos: Beginning-of-sequence token/vector
            eos: End-of-sequence token/vector
            max_len: Maximum generation length
            temperature: Sampling temperature
            num_generations: Number of sequences to generate in parallel

        Returns:
            List of generated sequence tensors
        """
        # Initialize: [batch_size, 1, ...] where ... is () for tokens or (5,) for vectors
        if bos.ndim == 0:  # Token case
            gen = bos.unsqueeze(0).expand(num_generations, 1)  # [batch, 1]
        else:  # Vector case
            gen = bos.unsqueeze(0).expand(num_generations, -1).unsqueeze(1)  # [batch, 1, 5]

        # Track which sequences have finished
        finished = torch.zeros(num_generations, dtype=torch.bool, device=self._device)

        # Initialize KV cache and position tracking
        kv_caches: list[tuple[Tensor, Tensor]] | None = None
        start_pos = 0

        # Process context once if it exists (prefill phase)
        if context is not None:
            # Expand context to match batch size: [batch, context_len, hidden_dim]
            context_expanded = context.unsqueeze(0).expand(num_generations, -1, -1)
            context_len = context.shape[0]

            # Process context through model with caching
            result = self._forward(context_expanded, start_pos=0, kv_caches=None, use_cache=True)
            match result:
                case (_, list() as caches):
                    kv_caches = caches
                case _:
                    raise ValueError(f"Unsupported result type: {type(result)}")
            start_pos = context_len

        for step in range(max_len):
            # Embed only the last generated token/vector
            if step == 0:
                # First step: embed the BOS token
                if gen.ndim == 2:  # Token case: [batch, 1]
                    gen_embed = output_embedder.embed(gen)  # [batch, 1, hidden_dim]
                else:  # Vector case: [batch, 1, 5]
                    gen_embed = output_embedder.embed(gen)  # [batch, 1, hidden_dim]
            else:
                # Subsequent steps: embed only the last token/vector
                if gen.ndim == 2:  # Token case
                    last_token = gen[:, -1:]  # [batch, 1]
                    gen_embed = output_embedder.embed(last_token)  # [batch, 1, hidden_dim]
                else:  # Vector case
                    last_vector = gen[:, -1:, :]  # [batch, 1, 5]
                    gen_embed = output_embedder.embed(last_vector)  # [batch, 1, hidden_dim]

            # Forward pass through model with caching
            result = self._forward(
                gen_embed, start_pos=start_pos, kv_caches=kv_caches, use_cache=True
            )
            match result:
                case (pred, list() as new_caches):
                    kv_caches = new_caches
                case _:
                    raise ValueError(f"Unsupported result type: {type(result)}")

            # Update position for next iteration
            start_pos += 1

            # Sample next token or vector based on embedder type
            if isinstance(output_embedder, VectorEmbedder):
                assert isinstance(pred, tuple)
                # Get predictions for last position: [batch, ...]
                last_pred = tuple(tensor[:, -1] for tensor in pred)
                assert len(last_pred) == 5
                # Sample vectors: [batch, 1, 5]
                next_vectors = self.sample_vector(last_pred, temperature=temperature)
                # Concatenate: [batch, seq_len+1, 5]
                gen = torch.cat([gen, next_vectors], dim=1)

                # Check for EOS (when any element of element-wise product equals 1.0)
                # next_vectors: [batch, 1, 5], eos: [5]
                eos_check = (next_vectors.squeeze(1) * eos.unsqueeze(0) == 1.0).any(dim=-1)
                finished = finished | eos_check
            else:
                assert isinstance(pred, Tensor)
                # Sample tokens: [batch, 1]
                next_tokens = self.sample_token(pred[:, -1], temperature=temperature)
                # Concatenate: [batch, seq_len+1]
                gen = torch.cat([gen, next_tokens], dim=1)

                # Check for EOS
                eos_check = next_tokens.squeeze(1) == eos
                finished = finished | eos_check

            # Early stopping if all sequences are done
            if finished.all():
                break

        # Convert batched tensor to list of individual sequences
        if gen.ndim == 2:  # Token case: [batch, seq_len]
            return [gen[i] for i in range(num_generations)]
        else:  # Vector case: [batch, seq_len, 5]
            return [gen[i] for i in range(num_generations)]
