import torch
from torch import Tensor

from ml_model.modules.embedder import MDNOutput


class SamplerMixin:
    def sample_token(self, logits: Tensor, temperature: float = 1.0) -> Tensor:
        """Sample token indices from logits.

        Args:
            logits: [..., vocab_size] unnormalized log probabilities
            temperature: sampling temperature (0.0 for greedy)

        Returns:
            [..., 1] sampled token indices
        """
        if temperature == 0.0:
            result = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            result = torch.multinomial(probs, num_samples=1)

        # Assert output has same number of dimensions as input
        assert result.ndim == logits.ndim, f"Expected {logits.ndim} dims, got {result.ndim} dims"
        return result

    def sample_vector(self, mdn_output: MDNOutput, temperature: float = 1.0) -> Tensor:
        """Sample handwriting vector from mixture density network output.

        Args:
            mdn_output: tuple of (mixtures, means, stds, rhos, pen_states)
            temperature: sampling temperature

        Returns:
            [batch_size, 1, 5] sampled vector (x, y, pen_up, pen_down, end_stroke)
        """
        mixtures, means, stds, rhos, pen_states = mdn_output

        selected_means, selected_stds, selected_rhos = self._sample_mixture(
            mixtures, means, stds, rhos, temperature
        )
        x, y = self._sample_xy(selected_means, selected_stds, selected_rhos, temperature)

        pen_up, pen_down, end_stroke = self._sample_pen_state(pen_states, temperature)

        next_vector = torch.stack([x, y, pen_up, pen_down, end_stroke], dim=-1).unsqueeze(1)
        return next_vector

    def _sample_mixture(
        self, mixtures: Tensor, means: Tensor, stds: Tensor, rhos: Tensor, temperature: float = 1.0
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = mixtures.size(0)
        batch_indices = torch.arange(batch_size)

        mixture_probs = torch.softmax(torch.log(mixtures) / temperature, dim=-1)
        mixture_indices = torch.multinomial(mixture_probs, 1).squeeze(-1)

        selected_means = means[batch_indices, mixture_indices]  # [batch_size, 2]
        selected_stds = stds[batch_indices, mixture_indices]  # [batch_size, 2]
        selected_rhos = rhos[batch_indices, mixture_indices]  # [batch_size]

        return selected_means, selected_stds, selected_rhos

    def _sample_xy(
        self, means: Tensor, stds: Tensor, rhos: Tensor, temperature: float = 1.0
    ) -> tuple[Tensor, Tensor]:
        batch_size = means.size(0)

        std_x, std_y = stds[:, 0] * temperature, stds[:, 1] * temperature
        z = torch.randn(batch_size, 2).to(means.device)

        x = means[:, 0] + std_x * z[:, 0]
        y = means[:, 1] + std_y * (rhos * z[:, 0] + torch.sqrt(1 - rhos**2) * z[:, 1])
        return x, y

    def _sample_pen_state(
        self, pen_states: Tensor, temperature: float = 1.0
    ) -> tuple[Tensor, Tensor, Tensor]:
        pen_probs = torch.softmax(pen_states / temperature, dim=-1)
        pen_state = torch.multinomial(pen_probs, 1).squeeze(-1)
        pen_up = (pen_state == 0).float()
        pen_down = (pen_state == 1).float()
        end_stroke = (pen_state == 2).float()
        return pen_up, pen_down, end_stroke
