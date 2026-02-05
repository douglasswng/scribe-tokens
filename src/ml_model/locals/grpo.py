import copy
from typing import Callable, Self, Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from constants import GRPO_EPSILON, GRPO_NUM_SAMPLES
from ink_repr.factory import ReprFactory
from ml_model.locals.htg import HTGModel
from ml_model.locals.local import LocalModel
from ml_model.metrics import compute_cer
from schemas.batch import Batch
from schemas.ink import DigitalInk
from schemas.instance import Instance

type HTRCallable = Callable[[Sequence[Instance]], Sequence[str]]


class GRPOModel(LocalModel):
    def __init__(self, htg_model: HTGModel, htr_callable: HTRCallable):
        super().__init__()
        self.htg_model = htg_model
        self.htr_callable = htr_callable
        self.num_samples = GRPO_NUM_SAMPLES

        # Create at eval mode and override train() to ensure stays in eval mode
        self.ref_model = copy.deepcopy(self.htg_model)

        # Freeze Reference Model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Create at eval mode and override train() to ensure stays in eval mode
        self.eval()

    def _update_instance(self, instance: Instance, ink: DigitalInk) -> Instance:
        updated_parsed = instance.parsed.model_copy(update={"ink": ink})
        repr = ReprFactory.from_ink(ink, instance.repr_id).to_tensor().to(self._device).detach()
        return Instance(
            parsed=updated_parsed,
            repr_id=instance.repr_id,
            repr=repr,
            char=instance.char,
        )

    def _generate_instances(self, instances: list[Instance]) -> list[Instance]:
        gen_inks = self.htg_model.batch_generate_inks(instances, num_generations=self.num_samples)
        gen_instances = [
            self._update_instance(orig_instance, gen_ink)
            for orig_instance, instance_gen_ink in zip(instances, gen_inks)
            for gen_ink in instance_gen_ink
        ]
        return gen_instances

    def _compute_batch_log_probs(
        self,
        instances: list[Instance],
        model: HTGModel,
    ) -> Tensor:
        """
        Compute log probabilities for all instances in a single batched forward pass.

        Args:
            instances: Flat list of instances to compute log probs for
            model: HTG model to use for computing log probs

        Returns:
            log_probs: [batch_size, max_seq_len] per-token log probs (0 for padding)
        """
        # Embed prompts and completions
        prompts = [model._char_embedder.embed(inst.char) for inst in instances]
        completion_inputs = [model._repr_embedder.embed(inst.repr_input) for inst in instances]

        # Calculate lengths
        prompt_lens = [p.shape[0] for p in prompts]
        completion_input_lens = [c.shape[0] for c in completion_inputs]
        max_completion_input_len = max(completion_input_lens)

        # Concatenate and pad
        inputs = [torch.cat([p, c], dim=0) for p, c in zip(prompts, completion_inputs)]
        input = pad_sequence(inputs, batch_first=True, padding_value=0)

        # Single forward pass
        pred = model._forward(input)

        # Compute per-token log probs
        batch_size = len(instances)
        log_probs = torch.zeros(
            batch_size, max_completion_input_len, device=self._device, dtype=torch.float
        )

        match pred:
            case Tensor():
                for i, (prompt_len, completion_len, inst) in enumerate(
                    zip(prompt_lens, completion_input_lens, instances)
                ):
                    logits = pred[i, prompt_len : prompt_len + completion_len]
                    target = inst.repr_target
                    log_probs[i, :completion_len] = self._categorical_log_prob(logits, target)

            case (mixtures, means, stds, rhos, pen_states):
                for i, (prompt_len, completion_len, inst) in enumerate(
                    zip(prompt_lens, completion_input_lens, instances)
                ):
                    target = inst.repr_target
                    x, y, pen = target[:, 0], target[:, 1], target[:, 2:]

                    coord_log_prob = self._bivariate_gaussian_log_prob(
                        mixtures[i, prompt_len : prompt_len + completion_len],
                        means[i, prompt_len : prompt_len + completion_len],
                        stds[i, prompt_len : prompt_len + completion_len],
                        rhos[i, prompt_len : prompt_len + completion_len],
                        x,
                        y,
                    )
                    pen_log_prob = self._categorical_log_prob(
                        pen_states[i, prompt_len : prompt_len + completion_len],
                        torch.argmax(pen, dim=-1),
                    )
                    log_probs[i, :completion_len] = coord_log_prob + pen_log_prob

            case _:
                raise ValueError(f"Unsupported prediction type: {type(pred)}")

        return log_probs

    def _losses(self, batch: Batch) -> dict[str, Tensor]:
        # Generate samples for each instance
        gen_instances = self._generate_instances(batch.instances)

        # Predicted text
        pred_texts = self.htr_callable(gen_instances)

        # Compute rewards
        rewards = [
            1 - compute_cer(pred_text, inst.parsed.text)
            for pred_text, inst in zip(pred_texts, gen_instances)
        ]

        # Compute batched log probs
        with torch.enable_grad():
            log_probs = self._compute_batch_log_probs(gen_instances, self.htg_model)
        with torch.no_grad():
            ref_log_probs = self._compute_batch_log_probs(gen_instances, self.ref_model)

        # Compute sequence mask for variable-length sequences
        target_lens = [inst.repr_target.shape[0] for inst in gen_instances]
        max_len = max(target_lens)
        mask = torch.zeros(len(gen_instances), max_len, device=self._device, dtype=torch.bool)
        for i, seq_len in enumerate(target_lens):
            mask[i, :seq_len] = True

        # Convert rewards to tensor and compute group-normalized advantages
        rewards = torch.tensor(rewards, device=self._device, dtype=torch.float)
        rewards_grouped = rewards.view(batch.size, self.num_samples)
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
        std_rewards = rewards_grouped.std(dim=1, keepdim=True) + GRPO_EPSILON
        advantages = ((rewards_grouped - mean_rewards) / std_rewards).flatten()

        # Log rewards
        if self.tracker is not None:
            self.tracker.log_metrics({"reward": rewards.mean().item()})
            self.tracker.log_metrics({"reward_std": std_rewards.mean().item()})

        # Compute GRPO loss with masking
        loss = self.grpo_loss(
            advantages.view(batch.size, self.num_samples),
            log_probs.view(batch.size, self.num_samples, -1),
            ref_log_probs.view(batch.size, self.num_samples, -1),
            mask.view(batch.size, self.num_samples, -1),
        )
        return {"grpo_loss": loss}

    def monitor(self, batch: Batch) -> None:
        instance = batch.get_random_instance()
        gen_ink = self.htg_model.batch_generate_inks([instance], num_generations=1)[0][0]
        gen_instance = self._update_instance(instance, gen_ink)
        pred_text = self.htr_callable([gen_instance])[0]
        self._track_ink(
            gen_ink, task="HTG_HTR", caption=f"True: {instance.parsed.text} | Pred: {pred_text}"
        )

    def validation_losses(self, batch: Batch) -> dict[str, Tensor]:
        """Skip expensive validation for GRPO - return empty metrics."""
        return {}

    def train(self, mode: bool = True) -> Self:
        # GRPO operates entirely in eval mode - always force eval on submodules
        super().train(False)
        return self

    def state_dict(self, *args, **kwargs):
        """
        Only save the trainable htg_model, not the frozen htr_model or ref_model.
        - htr_model is always loaded from pretrained HTR_SFT in factory
        - ref_model is always loaded from pretrained HTG in factory
        """
        return self.htg_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Load trained htg_model weights only.
        ref_model and htr_model remain unchanged (pretrained from factory).
        """
        return self.htg_model.load_state_dict(state_dict, strict=strict, assign=assign)
