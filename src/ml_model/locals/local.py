from abc import ABC, abstractmethod
from typing import Self, Sequence

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

    def validation_losses(self, batch: Batch) -> dict[str, Tensor]:
        """
        Compute losses for validation. Override to customize validation behavior.

        By default, uses the same losses as training. Models can override to:
        - Return empty dict to skip validation metrics
        - Return lightweight/approximate metrics
        - Disable expensive operations (e.g., sampling, generation)
        """
        return self._losses(batch)

    def _forward(
        self,
        input: Tensor,
        start_pos: int | Tensor = 0,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        attn_mask: Tensor | None = None,
        use_cache: bool = False,
    ) -> ForwardOutput | tuple[ForwardOutput, KVCaches]:
        """
        Forward pass through the model with KV cache support for efficient generation.

        This method supports both training (full sequence processing) and generation (incremental
        decoding with KV caching) modes.

        Args:
            input: Embedded input tensor
                Shape: (batch_size, seq_len, d_model)
                - Training mode: seq_len is the full sequence length
                - Generation mode with KV cache: seq_len=1 (only the newest token)

            start_pos: Position information for Rotary Position Embedding (RoPE)
                - int: Global start position. All sequences use consecutive positions
                  [start_pos, start_pos + seq_len). Common during generation where it
                  increments with each new token (0, 1, 2, ...).
                - Tensor: Shape (batch_size, seq_len). Explicit position index for each
                  element in the sequence. Used for non-consecutive position patterns.
                Default: 0

            kv_caches: Key-Value cache for efficient generation
                - None: No caching (typical during training or first generation step)
                - list[tuple[Tensor, Tensor]]: Cached keys and values from previous steps,
                  one tuple (k_cache, v_cache) per transformer layer
                  - k_cache shape: (batch_size, n_heads, cache_len, head_dim)
                  - v_cache shape: (batch_size, n_heads, cache_len, head_dim)
                  - cache_len: Number of previously processed tokens
                Default: None

            attn_mask: Attention mask to prevent attention to certain positions
                Shape: (batch_size, 1, seq_len, seq_len) or (1, 1, seq_len, seq_len)
                Convention: True/1 means "MASK IN" (position CAN be attended to)
                Default: None (only causal masking is applied)

            use_cache: Whether to return updated KV caches for the next generation step
                - False: Return only the model output (typical during training)
                - True: Return both output and updated KV caches (typical during generation)
                Default: False

        Returns:
            When use_cache=False:
                ForwardOutput: Model predictions
                    - Tensor: Logits for token prediction, shape (batch_size, seq_len, vocab_size)
                    - MDNOutput: Tuple of 5 tensors for mixture density network (vector prediction)

            When use_cache=True:
                tuple[ForwardOutput, KVCaches]: Model predictions and updated caches
                    - ForwardOutput: Same as above
                    - KVCaches: Updated key-value caches for all layers, same structure as
                      input kv_caches but with cache_len increased by seq_len

        Examples:
            Training (full sequence):
                >>> pred = model._forward(input)  # input: (32, 100, 512)
                >>> # pred: (32, 100, vocab_size)

            Generation (with KV cache):
                >>> # First step (prefill context)
                >>> output, caches = model._forward(context, use_cache=True)
                >>> # context: (1, 50, 512), caches[0][0]: (1, 8, 50, 64)
                >>>
                >>> # Subsequent steps (incremental decoding)
                >>> output, caches = model._forward(
                ...     next_token_emb,  # (1, 1, 512)
                ...     start_pos=50,
                ...     kv_caches=caches,
                ...     use_cache=True
                ... )
                >>> # caches[0][0] now: (1, 8, 51, 64)
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
        prompt_embedder: Embedder | None,
        completion_embedder: Embedder,
        prompt_attr: str | None,
        completion_input_attr: str,
        completion_target_attr: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Prepares batch tensors for training by concatenating prompt and completion sequences.

        The prompt part comes first and has no loss computed (loss_mask=False).
        The completion part comes second and has loss computed (loss_mask=True).

        Args:
            batch: Batch of instances
            prompt_embedder: Embedder for prompt part (e.g., char_embedder).
                If None, no prompt is used.
            completion_embedder: Embedder for completion part (e.g., repr_embedder)
            prompt_attr: Attribute name for prompt (e.g., "char").
                If None, no prompt is used.
            completion_input_attr: Attribute name for completion input (e.g., "repr_input")
            completion_target_attr: Attribute name for completion targets
                (e.g., "repr_target")

        Returns:
            Tuple of (padded_input, padded_target, padded_loss_mask)
        """
        inputs, targets, loss_masks = [], [], []

        for inst in batch.instances:
            # Embed completion sequences
            completion_input = completion_embedder.embed(getattr(inst, completion_input_attr))
            completion_target = getattr(inst, completion_target_attr)

            # Handle prompt if provided
            if prompt_embedder is not None and prompt_attr is not None:
                # Embed prompt
                prompt = prompt_embedder.embed(getattr(inst, prompt_attr))

                # Concatenate prompt + completion for input
                inputs.append(torch.cat([prompt, completion_input], dim=0))

                # Create dummy target for prompt (1D for tokens, 2D for vectors)
                if completion_target.ndim == 1:
                    dummy = torch.zeros(prompt.shape[0], dtype=torch.long, device=self._device)
                else:
                    dummy = torch.zeros(
                        prompt.shape[0], completion_target.shape[-1], device=self._device
                    )
                targets.append(torch.cat([dummy, completion_target], dim=0))

                # Create loss mask (False for prompt, True for completion)
                prompt_mask = torch.zeros(prompt.shape[0], dtype=torch.bool, device=self._device)
                completion_mask = torch.ones(
                    completion_input.shape[0], dtype=torch.bool, device=self._device
                )
                loss_masks.append(torch.cat([prompt_mask, completion_mask], dim=0))
            else:
                # No prompt, just use completion
                inputs.append(completion_input)
                targets.append(completion_target)
                loss_masks.append(
                    torch.ones(completion_input.shape[0], dtype=torch.bool, device=self._device)
                )

        # Pad sequences
        input = pad_sequence(inputs, batch_first=True, padding_value=0)
        target = pad_sequence(targets, batch_first=True, padding_value=0)
        loss_mask = pad_sequence(loss_masks, batch_first=True, padding_value=0)

        return input, target, loss_mask

    def _generate_sequences(
        self,
        context: Sequence[Tensor | None],
        bos: Tensor,
        eos: Tensor,
        output_embedder: Embedder,
        max_len: int,
        temperature: float,
        num_generations: int,
    ) -> list[list[Tensor]]:
        """
        Batched sequence generation with left-padding for parallel processing.

        Args:
            context: Sequence of context tensors (some may be None)
            bos: Beginning-of-sequence token/vector
            eos: End-of-sequence token/vector
            output_embedder: Embedder for the output sequence
            max_len: Maximum generation length
            temperature: Sampling temperature
            num_generations: Number of sequences to generate per context

        Returns:
            List of lists - outer list per context, inner list per generation
        """
        num_contexts = len(context)
        batch_size = num_contexts * num_generations

        # Determine context lengths (0 for None contexts)
        context_lens = torch.tensor(
            [c.shape[0] if c is not None else 0 for c in context], device=self._device
        )
        max_context_len = int(context_lens.max().item())

        # Get hidden dimension and dtype from first non-None context or embedder
        first_context = next((c for c in context if c is not None), None)
        if first_context is not None:
            hidden_dim = first_context.shape[-1]
            dtype = first_context.dtype
        else:
            dummy = output_embedder.embed(bos.unsqueeze(0).unsqueeze(0))
            hidden_dim = dummy.shape[-1]
            dtype = dummy.dtype

        # Initialize KV cache and masks
        kv_caches: list[tuple[Tensor, Tensor]] | None = None
        attention_mask: Tensor | None = None

        # Track per-sequence generation start positions (after their actual context)
        gen_start_positions = context_lens.repeat_interleave(num_generations)  # [batch_size]

        # Process context (prefill phase) if any context exists
        if max_context_len > 0:
            # Left-pad contexts: [num_contexts, max_context_len, hidden_dim]
            padded_contexts = torch.zeros(
                num_contexts, max_context_len, hidden_dim, device=self._device, dtype=dtype
            )

            attention_mask = torch.zeros(
                num_contexts, max_context_len, dtype=torch.bool, device=self._device
            )
            position_ids = torch.zeros(
                num_contexts, max_context_len, dtype=torch.long, device=self._device
            )

            for i, (c, c_len) in enumerate(zip(context, context_lens.tolist())):
                if c is not None and c_len > 0:
                    offset = max_context_len - c_len
                    padded_contexts[i, offset:] = c
                    attention_mask[i, offset:] = True
                    position_ids[i, offset:] = torch.arange(c_len, device=self._device)

            # Expand for num_generations
            padded_contexts = padded_contexts.repeat_interleave(num_generations, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_generations, dim=0)
            position_ids = position_ids.repeat_interleave(num_generations, dim=0)

            # Prefill: process all contexts through model
            key_attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).contiguous()
            result = self._forward(
                padded_contexts,
                start_pos=position_ids,
                kv_caches=None,
                attn_mask=key_attn_mask,
                use_cache=True,
            )
            match result:
                case (_, list() as caches):
                    kv_caches = caches
                case _:
                    raise ValueError(f"Unsupported result type: {type(result)}")

        # Initialize generation tensor and tracking
        gen = (
            bos.unsqueeze(0).expand(batch_size, 1)
            if bos.ndim == 0
            else bos.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self._device)

        # Generation loop
        for step in range(max_len):
            # Embed last generated token/vector
            last_item = gen[:, -1:] if gen.ndim == 2 else gen[:, -1:, :]
            gen_embed = output_embedder.embed(last_item)

            # Build attention mask: context padding + all generated tokens
            if attention_mask is not None and max_context_len > 0:
                context_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                gen_mask = torch.ones(
                    batch_size, 1, 1, step + 1, dtype=torch.bool, device=self._device
                )
                gen_attn_mask = torch.cat([context_mask, gen_mask], dim=-1).contiguous()
            else:
                gen_attn_mask = None

            # Forward pass with caching
            step_positions = (gen_start_positions + step).unsqueeze(1)
            result = self._forward(
                gen_embed,
                start_pos=step_positions,
                kv_caches=kv_caches,
                attn_mask=gen_attn_mask,
                use_cache=True,
            )
            match result:
                case (pred, list() as new_caches):
                    kv_caches = new_caches
                case _:
                    raise ValueError(f"Unsupported result type: {type(result)}")

            # Sample and check EOS
            if isinstance(output_embedder, VectorEmbedder):
                last_pred = tuple(tensor[:, -1] for tensor in pred)
                assert isinstance(last_pred, tuple) and len(last_pred) == 5
                next_item = self.sample_vector(last_pred, temperature=temperature)
                eos_check = (next_item.squeeze(1) * eos.unsqueeze(0) == 1.0).any(dim=-1)
            else:
                assert isinstance(pred, Tensor)
                next_item = self.sample_token(pred[:, -1], temperature=temperature)
                eos_check = next_item.squeeze(1) == eos

            gen = torch.cat([gen, next_item], dim=1)
            finished = finished | eos_check

            if finished.all():
                break

        # Reshape: [num_contexts * num_generations, ...] -> list[list[Tensor]]
        return [
            [gen[i * num_generations + j] for j in range(num_generations)]
            for i in range(num_contexts)
        ]
