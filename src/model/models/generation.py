from functools import partial

import torch
from core.constants import HIDDEN_DIM
from torch import Tensor

from core.data_schema import Batch, DigitalInk, Instance, PairBatch
from core.model import LocalModel, ModelId, Tracker
from model.models.batch_utils import BatchPreper
from model.models.loss_mixin import LossMixin
from model.modules.decoder import TransformerDecoder
from model.modules.embedder import CharEmbedder, Embedder, MDNOutput
from repr.factory import DefaultReprFactory


class GenerationModel(LocalModel, LossMixin):
    def __init__(
        self, model_id: ModelId, repr_embedder: Embedder, decoder: TransformerDecoder | None = None
    ):
        super().__init__()
        self._repr_embedder = repr_embedder
        self._char_embedder = CharEmbedder()

        self._decoder = decoder or TransformerDecoder()

        self._model_id = model_id
        self._ink_callable = partial(DefaultReprFactory.tensor_to_ink, id=self._model_id.repr_id)
        self._batch_preper = BatchPreper(
            task=model_id.task, repr_embedder=repr_embedder, char_embedder=self._char_embedder
        )

    def _forward(self, input: Tensor) -> Tensor | MDNOutput:
        pred = self._decoder(input)
        pred = self._repr_embedder.unembed(pred)
        return pred

    def losses(self, batch: Batch) -> dict[str, Tensor]:
        input, target, target_mask = self._batch_preper.prepare_batch(batch)
        pred = self._forward(input)
        match pred:
            case Tensor():
                loss = self.ce_loss(pred, target, target_mask)
                return {"ce": loss}
            case tuple():
                loss = self.nll_loss(pred, target, target_mask)
                return {"nll": loss}
            case _:
                raise ValueError(f"Invalid prediction type: {type(pred)}")

    def _sample_next_token(self, pred: Tensor, temperature: float = 1.0) -> Tensor:
        last_pred = pred[:, -1]
        probs = torch.softmax(last_pred / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

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

    def _sample_next_vector(self, pred: MDNOutput, temperature: float = 1.0) -> Tensor:
        last_pred = tuple(tensor[:, -1] for tensor in pred)
        mixtures, means, stds, rhos, pen_states = last_pred

        selected_means, selected_stds, selected_rhos = self._sample_mixture(
            mixtures, means, stds, rhos, temperature
        )
        x, y = self._sample_xy(selected_means, selected_stds, selected_rhos, temperature)

        pen_up, pen_down, end_stroke = self._sample_pen_state(pen_states, temperature)

        next_vector = torch.stack([x, y, pen_up, pen_down, end_stroke], dim=-1).unsqueeze(1)
        return next_vector

    def _generate_next_tensor(
        self, static_input: Tensor, gen_tensors: Tensor, temperature: float = 1.0
    ) -> Tensor:
        gen_embedded = self._repr_embedder.embed(gen_tensors)
        current_input = torch.cat([static_input, gen_embedded], dim=1)
        pred = self._forward(current_input)
        match pred:
            case Tensor():
                return self._sample_next_token(pred, temperature)
            case tuple():
                return self._sample_next_vector(pred, temperature)
            case _:
                raise ValueError(f"Invalid prediction type: {type(pred)}")

    def _terminate_generation(self, gen_tensors: Tensor, eos: Tensor) -> bool:
        has_eos = torch.any(gen_tensors == eos, dim=1)
        return bool(has_eos.all())

    def generate_inks(
        self,
        main_instance: Instance,
        ref_instance: Instance | None = None,
        num_gen: int = 1,
        temperature: float = 1.0,
        max_len: int = 500,
    ) -> list[DigitalInk]:
        if self.training:
            raise ValueError("Generation is not supported in training mode")

        with torch.no_grad():
            if ref_instance is not None:  # conditional generation
                ref_repr_embedded = self._repr_embedder.embed(ref_instance.repr)
                ref_char_embedded = self._char_embedder.embed(ref_instance.char)
                main_char_embedded = self._char_embedder.embed(main_instance.char)
                static_input = torch.cat(
                    [ref_repr_embedded, ref_char_embedded, main_char_embedded], dim=0
                )
                static_input = static_input.unsqueeze(0).expand(num_gen, -1, -1)
            else:  # unconditional generation
                device = main_instance.repr.device
                static_input = torch.empty(num_gen, 0, HIDDEN_DIM).to(device)

            if self._model_id.repr_id.is_token:
                gen_tensors = main_instance.repr_bos.unsqueeze(0).expand(num_gen, -1)
            else:
                gen_tensors = (
                    main_instance.repr_bos.unsqueeze(0).unsqueeze(0).expand(num_gen, -1, -1)
                )

            for _ in range(max_len):
                next_tensor = self._generate_next_tensor(
                    static_input=static_input, gen_tensors=gen_tensors, temperature=temperature
                )

                gen_tensors = torch.cat([gen_tensors, next_tensor], dim=1)

                if self._terminate_generation(gen_tensors, main_instance.repr_eos):
                    break

            inks = [self._ink_callable(tensor=gen_tensor) for gen_tensor in gen_tensors]
            return inks

    def monitor(self, batch: Batch, tracker: Tracker | None = None) -> None:
        assert isinstance(batch, PairBatch)
        main_instance, ref_instance = batch.get_random_instance_pair()
        gen_ink = self.generate_inks(main_instance=main_instance, ref_instance=ref_instance)[0]
        main_text = main_instance.parsed.text
        self._monitor_ink(gen_ink, "Generated", main_text, tracker)

    def _monitor_ink(
        self, ink: DigitalInk, task: str, caption: str, tracker: Tracker | None = None
    ) -> None:
        ink.visualise(name=f"{task}: {caption}")
        if tracker is not None:
            tracker.log_image(ink.to_image(), task, caption)


if __name__ == "__main__":
    from core.utils import distributed_context

    from core.model import ModelId, Task
    from dataloader.create import create_dataloaders
    from model.factory import ReprEmbedderFactory

    for model_id in ModelId.create_task_model_ids(Task.GENERATION)[::]:
        print(model_id)
        train_loader, val_loader, test_loader = create_dataloaders(
            model_id=model_id,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        repr_embedder = ReprEmbedderFactory.create(model_id.repr_id)
        model = GenerationModel(model_id=model_id, repr_embedder=repr_embedder).to(
            distributed_context.device
        )
        for batch in train_loader:
            model.train()
            losses = model.losses(batch)
            print(losses)
            model.eval()
            model.monitor(batch)
            break
