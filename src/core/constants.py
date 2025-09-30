from pathlib import Path


# Paths
BASE_DIR = Path(__file__).parent.parent.parent

DATA_DIR = BASE_DIR / "data"
TMP_DIR = BASE_DIR / "tmp"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
SPLIT_DIR = DATA_DIR / "split"

RAW_IAM_DIR = RAW_DIR / "iam"
RAW_IAM_SPLIT_DIR = RAW_IAM_DIR / "split"
PARSED_IAM_DIR = PARSED_DIR / "iam"
SPLIT_IAM_DIR = SPLIT_DIR / "iam"

TOKENISERS_DIR = ARTIFACTS_DIR / "tokenisers"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
MODELS_DIR = ARTIFACTS_DIR / "models"
TRACKERS_DIR = ARTIFACTS_DIR / "trackers"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

# Dataset statistics
CHARS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
         + "abcdefghijklmnopqrstuvwxyz"
         + "0123456789"
         + " !\"#%&'()+,-./:;?")
NUM_CHARS = len(CHARS)  # 79

# Experiment hyperparameters
VOCAB_SIZE = 32000
DELTA = 8
MAX_LEN = int(1e4)
SCRIBE_DOWNSAMPLE_FACTOR = 16 // DELTA  # for scribe's post processor

# Model hyperparameters
HIDDEN_DIM = 384
FFN_FACTOR = 8/3  # use swiglu ffn
NUM_LAYERS = 12
NUM_HEADS = 6
DROPOUT = 0.1
NUM_MIXTURES = 20

# Training hyperparameters
UNKNOWN_TOKEN_RATE = 0.002  # match the unknown rate on the validation set (~0.2% when DELTA=8, ~0.3% when DELTA=4)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3  # didnt seem to need learning rate warmup or cosine decay
NUM_EPOCHS = 300
PATIENCE = int(0.1 * NUM_EPOCHS)  # 10% of the epochs
WEIGHT_DECAY = 0.1

# Augmenter hyperparameters
SCALE_RANGE = 0.3  # scale factor between (1 - SCALE_RANGE, 1 + SCALE_RANGE)
SHEAR_FACTOR = 0.5  # shear factor between (-SHEAR_FACTOR, SHEAR_FACTOR)
ROTATE_ANGLE = 5  # rotate angle (degrees) between (-ROTATE_ANGLE, ROTATE_ANGLE)
JITTER_SIGMA = 5  # the std of gaussian noise added to the points
AUGMENT_PROB = 0.5  # each augmentation has this probability of being applied, independently