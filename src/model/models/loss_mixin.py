from torch import Tensor
import torch.nn as nn


class LossMixin:
    def ce_loss(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        logits_flat = pred.reshape(-1, pred.size(-1))
        target_flat = target.reshape(-1)
        mask_flat = mask.reshape(-1)
        
        valid_mask = mask_flat.bool()
        valid_logits = logits_flat[valid_mask]
        valid_targets = target_flat[valid_mask]

        criterion = nn.CrossEntropyLoss(reduction='none')
        loss: Tensor = criterion(valid_logits, valid_targets)
        return loss.mean()