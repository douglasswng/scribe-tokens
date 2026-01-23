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

        # Store reference model state for KL penalty
        self._ref_state = {k: v.clone() for k, v in self.htg_model.state_dict().items()}

    @torch.no_grad()
    def _generate_sample(self, instance: Instance) -> tuple[Tensor, str]:
        """
        Generate a single sample from HWG model.
        Returns the generated representation sequence and its text prediction from HWR.
        """
        # Use HWG model's generate_ink method
        ink = self.htg_model.generate_ink(instance)

        # Convert ink back to tensor for generation sequence
        gen_repr = ReprFactory.from_ink(ink, repr_id=instance.repr_id).to_tensor()

        # Create a temporary instance for HWR prediction
        temp_instance = Instance(
            parsed=instance.parsed,
            repr_id=instance.repr_id,
            repr=gen_repr,
            char=instance.char,
        )
        pred_text = self.htr_model.predict_text(temp_instance)

        return gen_repr, pred_text

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
        # Temporarily swap to reference weights if needed
        if use_ref_model:
            current_state = self.htg_model.state_dict()
            self.htg_model.load_state_dict(self._ref_state)

        # Embed context and target sequence
        context_embed = self.htg_model._char_embedder.embed(instance.char).unsqueeze(0)
        context_len = len(context_embed[0])
        target_embed = self.htg_model._repr_embedder.embed(gen_sequence).unsqueeze(0)

        # Concatenate and forward pass
        input = torch.cat([context_embed, target_embed], dim=1)
        pred = self.htg_model._forward(input)

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
            case tuple():
                # MDN-based: use nll_loss
                mixtures, means, stds, rhos, pen_states = pred
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

        # Restore current weights if we used reference model
        if use_ref_model:
            self.htg_model.load_state_dict(current_state)

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

            # Generate multiple samples
            rewards = []
            log_probs = []

            for _ in range(self.num_samples):
                gen_seq, pred_text = self._generate_sample(instance)
                cer = compute_cer(pred_text, target_text)
                reward = 1.0 - cer  # Higher reward for lower CER

                # Compute log probs under current policy
                log_prob = self._compute_log_prob(instance, gen_seq, use_ref_model=False)

                rewards.append(reward)
                log_probs.append(log_prob)

                # Compute log probs under reference policy for KL penalty
                ref_log_prob = self._compute_log_prob(instance, gen_seq, use_ref_model=True)
                all_ref_log_probs.append(ref_log_prob)

            # Accumulate across all instances
            all_rewards.extend(rewards)
            all_log_probs.extend(log_probs)

        # Compute GRPO loss using LossMixin method
        grpo_loss, avg_reward = self.grpo_loss(all_rewards, all_log_probs, all_ref_log_probs)

        return {
            "grpo_loss": grpo_loss,
            "avg_reward": avg_reward,
        }

    def monitor(self, batch: Batch) -> None:
        return self.htg_model.monitor(batch)


if __name__ == "__main__":
    from dataloader.create import create_dataloaders
    from ml_model.factory import ModelFactory
    from ml_model.id import ModelId, Task
    from utils.distributed_context import distributed_context

    for model_id in ModelId.create_task_model_ids(Task.HTG_GRPO)[:]:
        print(f"Testing GRPO model: {model_id}")
        train_loader, val_loader, test_loader = create_dataloaders(
            model_id=model_id,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        model = ModelFactory.create(model_id).to(distributed_context.device)
        print(f"Model has {float(model.num_params) / 1e6:.2f}M params")

        for batch in train_loader:
            model.train()
            print("Computing GRPO loss...")
            losses = model(batch)
            print(f"Losses: {losses}")
            model.eval()
            print("Monitoring...")
            model.monitor(batch)
            break
