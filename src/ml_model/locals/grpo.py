import copy
from typing import Self

import torch
from torch import Tensor

from constants import GRPO_NUM_SAMPLES
from ink_repr.factory import ReprFactory
from ml_model.locals.htg import HTGModel
from ml_model.locals.htr import HTRModel
from ml_model.locals.local import LocalModel
from ml_model.metrics import compute_cer
from schemas.batch import Batch
from schemas.instance import Instance


class GRPOModel(LocalModel):
    def __init__(self, htg_model: HTGModel, htr_model: HTRModel):
        super().__init__()
        self.htg_model = htg_model
        self.htr_model = htr_model
        self.num_samples = GRPO_NUM_SAMPLES

        # Freeze HWR model parameters
        for param in self.htr_model.parameters():
            param.requires_grad = False
        self.htr_model.eval()

        # Create a separate Reference Model (Deep Copy)
        # We assume htg_model can be deepcopied. If not, re-instantiate it using factory logic.
        self.ref_model = copy.deepcopy(self.htg_model)

        # Freeze Reference Model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

    @torch.no_grad()
    def _generate_samples(self, instance: Instance, num_samples: int) -> list[tuple[Tensor, str]]:
        """
        Generate multiple samples from HTG model for a single instance.
        Returns list of (generated representation sequence, predicted text) tuples.
        """
        # Use HTG model's generate_inks method to generate all samples at once
        inks = self.htg_model.generate_inks(instance, num_generations=num_samples)

        results = []
        for ink in inks:
            # Convert ink back to tensor for generation sequence
            gen_repr = ReprFactory.from_ink(ink, repr_id=instance.repr_id).to_tensor()

            # Create a temporary instance for HTR prediction
            temp_instance = Instance(
                parsed=instance.parsed,
                repr_id=instance.repr_id,
                repr=gen_repr,
                char=instance.char,
            )
            pred_text = self.htr_model.predict_text(temp_instance)
            results.append((gen_repr, pred_text))

        return results

    def _compute_log_prob(
        self, instance: Instance, gen_sequence: Tensor, use_ref_model: bool = False
    ) -> Tensor:
        """
        Compute log probability of the generated sequence.

        Args:
            instance: The input instance
            gen_sequence: The generated sequence
            use_ref_model: If True, use reference model weights for KL penalty

        Returns:
            Tensor: Sum of log probabilities over the sequence (scalar)
        """
        model = self.ref_model if use_ref_model else self.htg_model

        # Embed context and target sequence
        context_embed = model._char_embedder.embed(instance.char).unsqueeze(0)
        context_len = len(context_embed[0])
        target_embed = model._repr_embedder.embed(gen_sequence).unsqueeze(0)

        # Concatenate and forward pass
        input = torch.cat([context_embed, target_embed], dim=1)
        pred = model._forward(input)

        # Compute log probability based on prediction type using LossMixin methods
        seq_len = len(gen_sequence)
        mask = torch.ones(1, seq_len - 1, dtype=torch.bool, device=gen_sequence.device)

        match pred:
            case Tensor():
                # Token-based: use ce_loss
                logits = pred[:, context_len : context_len + seq_len - 1]
                target_ids = gen_sequence[1:].unsqueeze(0)
                ce = self.ce_loss(logits, target_ids, mask)
                log_prob = -ce * (seq_len - 1)
            case (mixtures, means, stds, rhos, pen_states):
                # MDN-based: use nll_loss
                mixtures_slice = mixtures[:, context_len : context_len + seq_len - 1]
                means_slice = means[:, context_len : context_len + seq_len - 1]
                stds_slice = stds[:, context_len : context_len + seq_len - 1]
                rhos_slice = rhos[:, context_len : context_len + seq_len - 1]
                pen_states_slice = pen_states[:, context_len : context_len + seq_len - 1]
                target_vecs = gen_sequence[1:].unsqueeze(0)
                pred_slice = (mixtures_slice, means_slice, stds_slice, rhos_slice, pen_states_slice)
                nll = self.nll_loss(pred_slice, target_vecs, mask)
                log_prob = -nll * (seq_len - 1)
            case _:
                raise ValueError(f"Unsupported prediction type: {type(pred)}")

        return log_prob

    def _losses(self, batch: Batch) -> dict[str, Tensor]:
        """
        Compute GRPO loss:
        1. Generate multiple samples per instance
        2. Evaluate each sample with HTR (compute CER)
        3. Compute group advantages (reward - mean_group_reward)
        4. Compute policy gradient loss with advantages
        5. Optionally add KL penalty with reference policy
        """
        all_rewards = []
        all_log_probs = []
        all_ref_log_probs = []

        for instance in batch.instances:
            target_text = instance.parsed.text

            # Generate samples
            samples = self._generate_samples(instance, num_samples=self.num_samples)
            gen_seqs = [sample[0] for sample in samples]
            pred_texts = [sample[1] for sample in samples]

            # Calculate rewards
            cers = [compute_cer(pred_text, target_text) for pred_text in pred_texts]
            rewards = [1.0 - cer for cer in cers]

            # Calculate log probabilities
            log_probs = [
                self._compute_log_prob(instance, gen_seq, use_ref_model=False)
                for gen_seq in gen_seqs
            ]
            ref_log_probs = [
                self._compute_log_prob(instance, gen_seq, use_ref_model=True)
                for gen_seq in gen_seqs
            ]

            # Accumulate across all instances
            all_rewards.extend(rewards)
            all_log_probs.extend(log_probs)
            all_ref_log_probs.extend(ref_log_probs)

        # Compute GRPO loss using LossMixin method
        grpo_loss, avg_reward = self.grpo_loss(all_rewards, all_log_probs, all_ref_log_probs)

        return {
            "grpo_loss": grpo_loss,
            "avg_reward": avg_reward,
        }

    def monitor(self, batch: Batch) -> None:
        return self.htg_model.monitor(batch)

    def train(self, mode: bool = True) -> Self:
        """
        Override train mode to ensure reference and reward models
        remain in eval mode (frozen).
        """
        super().train(mode)

        # Explicitly force these back to eval mode
        self.htr_model.eval()
        self.ref_model.eval()

        return self
