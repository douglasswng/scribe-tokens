from pathlib import Path


# Paths
BASE_DIR = Path(__file__).parent.parent.parent

DATA_DIR = BASE_DIR / "data"
TMP_DIR = BASE_DIR / "tmp"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"

RAW_IAM_DIR = RAW_DIR / "iam"
PARSED_IAM_DIR = PARSED_DIR / "iam"

TOKENISERS_DIR = ARTIFACTS_DIR / "tokenisers"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
MODELS_DIR = ARTIFACTS_DIR / "models"
TRACKERS_DIR = ARTIFACTS_DIR / "trackers"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

# Dataset statistics
WRITERS = list([str(id) for id in range(10027, 10222)])
NUM_WRITERS = len(WRITERS)  # 195

CHARS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
         + "abcdefghijklmnopqrstuvwxyz"
         + "0123456789"
         + " !\"#%&'()+,-./:;?")
NUM_CHARS = len(CHARS)  # 79

# Experiment hyperparameters
VOCAB_SIZE = 25000
DELTA = 2
MAX_LEN = int(1e5)
SCRIBE_DOWNSAMPLE_FACTOR = 16 // DELTA

# Model hyperparameters
HIDDEN_DIM = 256
FFN_FACTOR = 4
NUM_LSTM_LAYERS = 2
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
NUM_HEADS = 4
DROPOUT = 0.1
NUM_MIXTURES = 20

# Training hyperparameters
UNKNOWN_TOKEN_RATE = 0.002  # match the unknown rate on the validation set
BATCH_SIZE = 64
LEARNING_RATE = 1e-3  # didnt seem to need learning rate warmup or cosine decay
VECTOR_LOSS_FACTOR = 0.1  # multiply vector loss by this factor to balance with token loss
NUM_EPOCHS = 100
PATIENCE = int(0.1 * NUM_EPOCHS)  # 10% of the epochs
WEIGHT_DECAY = 1e-2

# Augmenter hyperparameters
SCALE_RANGE = 0.2  # scale factor between (1 - SCALE_RANGE, 1 + SCALE_RANGE)
SHEAR_FACTOR = 0.2  # shear factor between (-SHEAR_FACTOR, SHEAR_FACTOR)
ROTATE_ANGLE = 10  # rotate angle (degrees) between (-ROTATE_ANGLE, ROTATE_ANGLE)

# BERT-style pretraining hyperparameters
MASK_PROB = 0.15
MASK_TOKEN_RATE = 0.8
RANDOM_TOKEN_RATE = 0.1
SAME_TOKEN_RATE = 0.1