import torch
import torch.nn.functional as F
from torch import Tensor

from ml_model.modules.embedder import MDNOutput


class LossMixin:
    def _bivariate_gaussian_log_prob(
        self, mixtures: Tensor, means: Tensor, stds: Tensor, rhos: Tensor, x: Tensor, y: Tensor
    ) -> Tensor:
        """
        Compute log probabilities from a bivariate Gaussian mixture model.

        This is the core computation used by both loss calculation and policy gradient methods.
        Handles both batched and unbatched inputs flexibly.

        Args:
            mixtures: [..., num_mixtures] - mixture weights
            means: [..., num_mixtures, 2] - (x, y) means for each mixture
            stds: [..., num_mixtures, 2] - (x, y) standard deviations
            rhos: [..., num_mixtures] - correlation coefficients
            x: [...] - target x coordinates
            y: [...] - target y coordinates

        Returns:
            Tensor: Log probabilities with shape [...] (one per input position)
        """
        # Center the coordinates
        x_centered = x.unsqueeze(-1) - means[..., 0]
        y_centered = y.unsqueeze(-1) - means[..., 1]

        # Compute standardized terms
        x_term = (x_centered / stds[..., 0]) ** 2
        y_term = (y_centered / stds[..., 1]) ** 2
        xy_term = 2 * rhos * (x_centered / stds[..., 0]) * (y_centered / stds[..., 1])

        # Compute exponent term
        z = x_term + y_term - xy_term
        exp_term = -z / (2 * (1 - rhos**2))

        # Compute log probability for each mixture component
        norm_const = 2 * torch.pi * stds[..., 0] * stds[..., 1] * torch.sqrt(1 - rhos**2)
        log_probs = -torch.log(norm_const) + exp_term + torch.log(mixtures)

        # Sum over mixture components (logsumexp for numerical stability)
        return torch.logsumexp(log_probs, dim=-1)

    def _categorical_log_prob(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute log probabilities for categorical distributions.

        Used for both pen states and token predictions.

        Args:
            logits: [..., num_classes] - unnormalized logits
            targets: [...] - target class indices (long tensor)

        Returns:
            Tensor: Log probabilities with shape [...]
        """
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(dim=-1, index=targets.unsqueeze(-1).long()).squeeze(-1)

    def ce_loss(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Cross-entropy loss for token-based models."""
        logits_flat = pred.reshape(-1, pred.size(-1))
        target_flat = target.reshape(-1)
        mask_flat = mask.reshape(-1)

        if mask_flat.dtype != torch.bool:
            raise ValueError(f"Mask must be a boolean tensor, got {mask_flat.dtype}")

        valid_logits = logits_flat[mask_flat]
        valid_targets = target_flat[mask_flat]

        # Use unified categorical log prob computation
        log_probs = self._categorical_log_prob(valid_logits, valid_targets)
        return -log_probs.mean()

    def nll_loss(self, pred: MDNOutput, target: Tensor, mask: Tensor) -> Tensor:
        """Negative log-likelihood loss for vector-based MDN models."""
        mixtures, means, stds, rhos, pen_states = pred

        x, y, pen = target[:, :, 0], target[:, :, 1], target[:, :, 2:]

        # Compute log probabilities using unified methods
        coord_log_prob = self._bivariate_gaussian_log_prob(mixtures, means, stds, rhos, x, y)
        pen_log_prob = self._categorical_log_prob(pen_states, torch.argmax(pen, dim=-1))

        # Negative log likelihood (loss is negative log prob)
        total_log_prob = coord_log_prob + pen_log_prob
        masked_log_prob: Tensor = total_log_prob * mask

        return -masked_log_prob.sum() / mask.sum()

    def grpo_loss(  # TRL style
        self,
        advantages: Tensor,
        log_probs: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        GRPO loss with optional masking for variable-length sequences.

        Args:
            advantages: [batch_size, group_size] - per-sample advantages
            log_probs: [batch_size, group_size, seq_len] - per-token log probs from policy
            ref_log_probs: [batch_size, group_size, seq_len] - per-token log probs from reference
            mask: [batch_size, group_size, seq_len] - boolean mask (True = valid token)
        Returns:
            grpo_loss: [] - grpo loss tensor
        """
        advantages = advantages.unsqueeze(-1).expand_as(log_probs)
        policy_loss = -torch.exp(log_probs - log_probs.detach()) * advantages
        return (policy_loss * mask).sum() / mask.sum()
