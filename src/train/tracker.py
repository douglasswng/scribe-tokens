from core.train.tracker import MLFlowTracker, SwanLabTracker, Tracker
from core.model import ModelId
from core.utils import distributed_context
from core.constants import TRACKERS_DIR, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE, DELTA, HIDDEN_DIM, NUM_HEADS, DROPOUT, VOCAB_SIZE, FFN_FACTOR, NUM_LAYERS


class DefaultTracker(Tracker):
    def __init__(self,
                 tracker_name: str,
                 model_id: ModelId):
        if distributed_context.is_worker:
            return
        
        self.begin_experiment(name='ScribeTokens',
                              artifact_dir=TRACKERS_DIR / tracker_name)
        self.begin_run(tags=[model_id.task.value, str(model_id.repr_id)],
                       run_name=str(model_id))
        self.log_params({'batch_size': BATCH_SIZE,
                         'learning_rate': LEARNING_RATE,
                         'weight_decay': WEIGHT_DECAY,
                         'num_epochs': NUM_EPOCHS,
                         'patience': PATIENCE,
                         'delta': DELTA,
                         'hidden_dim': HIDDEN_DIM,
                         'num_layers': NUM_LAYERS,
                         'num_heads': NUM_HEADS,
                         'dropout': DROPOUT,
                         'vocab_size': VOCAB_SIZE,
                         'ffn_factor': FFN_FACTOR})


class DefaultMLFlowTracker(DefaultTracker, MLFlowTracker):
    def __init__(self, model_id: ModelId):
        MLFlowTracker.__init__(self)
        DefaultTracker.__init__(self, tracker_name='mlflow',
                                model_id=model_id)
    

class DefaultSwanLabTracker(DefaultTracker, SwanLabTracker):
    def __init__(self, model_id: ModelId):
        SwanLabTracker.__init__(self)
        DefaultTracker.__init__(self, tracker_name='swanlab',
                                model_id=model_id)