import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.modules.embedder import MDNOutput


class LossMixin:
    def ce_loss(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        logits_flat = pred.reshape(-1, pred.size(-1))
        target_flat = target.reshape(-1)
        mask_flat = mask.reshape(-1)

        if mask_flat.dtype != torch.bool:
            raise ValueError(f"Mask must be a boolean tensor, got {mask_flat.dtype}")

        valid_logits = logits_flat[mask_flat]
        valid_targets = target_flat[mask_flat]

        criterion = nn.CrossEntropyLoss(reduction="none")
        loss: Tensor = criterion(valid_logits, valid_targets)
        return loss.mean()

    def _coord_loss(
        self, mixtures: Tensor, means: Tensor, stds: Tensor, rhos: Tensor, x: Tensor, y: Tensor
    ) -> Tensor:
        x_centered = x.unsqueeze(-1) - means[:, :, :, 0]
        y_centered = y.unsqueeze(-1) - means[:, :, :, 1]

        x_term = (x_centered / stds[:, :, :, 0]) ** 2
        y_term = (y_centered / stds[:, :, :, 1]) ** 2
        xy_term = 2 * rhos * (x_centered / stds[:, :, :, 0]) * (y_centered / stds[:, :, :, 1])

        z = x_term + y_term - xy_term
        exp_term = -z / (2 * (1 - rhos**2))

        norm_const = 2 * torch.pi * stds[:, :, :, 0] * stds[:, :, :, 1] * torch.sqrt(1 - rhos**2)
        log_probs = -torch.log(norm_const) + exp_term
        log_probs = log_probs + torch.log(mixtures)

        coord_loss = -torch.logsumexp(log_probs, dim=-1)
        return coord_loss

    def _pen_loss(self, pen_states: Tensor, pen: Tensor) -> Tensor:
        pen_loss = F.cross_entropy(
            input=pen_states.transpose(-1, -2), target=torch.argmax(pen, dim=-1), reduction="none"
        )
        return pen_loss

    def nll_loss(self, pred: MDNOutput, target: Tensor, mask: Tensor) -> Tensor:
        mixtures, means, stds, rhos, pen_states = pred

        x, y, pen = target[:, :, 0], target[:, :, 1], target[:, :, 2:]

        coord_loss = self._coord_loss(mixtures, means, stds, rhos, x, y)
        pen_loss = self._pen_loss(pen_states, pen)

        total_loss = coord_loss + pen_loss
        masked_loss: Tensor = total_loss * mask

        return masked_loss.sum() / mask.sum()
