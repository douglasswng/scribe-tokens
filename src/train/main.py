from core.model import ModelId, ModelPaths
from train.trainer import DefaultTrainer
from core.constants import NUM_EPOCHS


def is_trained(model_id: ModelId) -> bool:
    model_paths = ModelPaths(model_id)
    return model_paths.model_path.exists()


def main() -> None:
    model_ids = ModelId.create_defaults()
    for model_id in model_ids:
        if is_trained(model_id):
            print(f"Model {model_id} already trained")
            continue
        
        trainer = DefaultTrainer(model_id)
        trainer.train_with_resume(NUM_EPOCHS)


if __name__ == "__main__":
    from core.utils import clear_folder
    from core.utils.distributed_context import distributed_context
    from core.constants import CHECKPOINTS_DIR, MODELS_DIR, TRACKERS_DIR
    
    if distributed_context.is_master:
        clear_folder(CHECKPOINTS_DIR, confirm=True)
        clear_folder(TRACKERS_DIR, confirm=False)
        #clear_folder(MODELS_DIR, confirm=True)

    distributed_context.barrier()
    main()