from pathlib import Path
from typing import Literal

from utils.clear_folder import clear_folder

# Paths
BASE_DIR = Path(__file__).parent.parent

TMP_DIR = BASE_DIR / "tmp"
clear_folder(TMP_DIR, confirm=False)

ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

DATASET: Literal["iam", "deepwriting"] = "deepwriting"
RAW_DIR = DATA_DIR / "raw" / DATASET
PARSED_DIR = DATA_DIR / "parsed" / DATASET
_SPLIT_DIR = DATA_DIR / "split" / DATASET
TRAIN_SPLIT_PATH = _SPLIT_DIR / "train.txt"
VAL_SPLIT_PATH = _SPLIT_DIR / "val.txt"
TEST_SPLIT_PATH = _SPLIT_DIR / "test.txt"

TOKENISERS_DIR = Path("tokenisers")
MODELS_DIR = Path("models")

CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
TRACKERS_DIR = ARTIFACTS_DIR / "trackers"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# Dataset statistics
CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    + "abcdefghijklmnopqrstuvwxyz"
    + "0123456789"
    + " !\"#%&'()+,-./:;?"
)
NUM_CHARS = len(CHARS)  # 79

# Experiment hyperparameters
VOCAB_SIZE = 32000
DELTA = 8
MAX_LEN = int(1e4)
SCRIBE_DOWNSAMPLE_FACTOR = 16 // DELTA  # for scribe's post processor

# Model hyperparameters
HIDDEN_DIM = 384
FFN_FACTOR = 8 / 3  # use swiglu ffn
NUM_LAYERS = 12
NUM_HEADS = 6
DROPOUT = 0.2
NUM_MIXTURES = 20

# Training hyperparameters
UNKNOWN_TOKEN_RATE = 0.004  # match the unknown rate on the validation set
BATCH_SIZE = 64
WEIGHT_DECAY = 0.1
LEARNING_RATE = 3e-4
NUM_EPOCHS = 100
PATIENCE_FACTOR = 0.1
GRAD_ACCUM_STEPS = 2

# Augmenter hyperparameters
SCALE_RANGE = 0.3  # scale factor between (1 - SCALE_RANGE, 1 + SCALE_RANGE)
SHEAR_FACTOR = 0.5  # shear factor between (-SHEAR_FACTOR, SHEAR_FACTOR)
ROTATE_ANGLE = 5  # rotate angle (degrees) between (-ROTATE_ANGLE, ROTATE_ANGLE)
JITTER_SIGMA = 5  # the std of gaussian noise added to the points
AUGMENT_PROB = 0.5  # each augmentation has this probability of being applied, independently

# GRPO hyperparameters
GRPO_NUM_SAMPLES = 8  # number of samples to generate per instance for GRPO
LEARNING_RATE = 3e-5
GRAD_ACCUM_STEPS = 2
GRPO_BETA = 1e-3

GRPO_EPSILON = 1e-4  # stabilise division by std in advtange calculation
GRPO_STEPS = 500  # no concept of epochs for GRPO (30 seconds per step on BH200)
GRPO_MONITOR_EVERY = 10

# Vector Repr Stability hyperparameters
INK_SCALE = 0.1
MDN_STD_MIN = 0.1  # minimum standard deviation for MDN stability
MDN_RHO_MAX = 0.99  # maximum correlation coefficient for MDN stability
