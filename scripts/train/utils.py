"""Utility functions for training scripts."""

from typing import Any

from constants import (
    AUGMENT_PROB,
    BATCH_SIZE,
    DELTA,
    DROPOUT,
    FFN_FACTOR,
    HIDDEN_DIM,
    INK_SCALE,
    JITTER_SIGMA,
    LEARNING_RATE,
    MAX_LEN,
    MDN_RHO_MAX,
    MDN_STD_MIN,
    NUM_CHARS,
    NUM_EPOCHS,
    NUM_HEADS,
    NUM_LAYERS,
    NUM_MIXTURES,
    PATIENCE_FACTOR,
    ROTATE_ANGLE,
    SCALE_RANGE,
    SCRIBE_DOWNSAMPLE_FACTOR,
    SHEAR_FACTOR,
    UNKNOWN_TOKEN_RATE,
    VOCAB_SIZE,
    WEIGHT_DECAY,
)


def get_params_dict() -> dict[str, Any]:
    """Get a dictionary of all important training hyperparameters and constants.

    Returns a comprehensive dict of parameters for experiment tracking, including:
    - Model architecture hyperparameters
    - Training hyperparameters
    - Tokenization settings
    - Data augmentation settings
    - GRPO-specific settings
    - Vector representation stability settings
    """
    return {
        # Model architecture
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "dropout": DROPOUT,
        "ffn_factor": FFN_FACTOR,
        "num_mixtures": NUM_MIXTURES,
        # Training hyperparameters
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "unknown_token_rate": UNKNOWN_TOKEN_RATE,
        "patience_factor": PATIENCE_FACTOR,
        # Tokenization
        "vocab_size": VOCAB_SIZE,
        "delta": DELTA,
        "max_len": MAX_LEN,
        "num_chars": NUM_CHARS,
        "scribe_downsample_factor": SCRIBE_DOWNSAMPLE_FACTOR,
        # Data augmentation
        "scale_range": SCALE_RANGE,
        "shear_factor": SHEAR_FACTOR,
        "rotate_angle": ROTATE_ANGLE,
        "jitter_sigma": JITTER_SIGMA,
        "augment_prob": AUGMENT_PROB,
        # Vector representation stability
        "ink_scale": INK_SCALE,
        "mdn_std_min": MDN_STD_MIN,
        "mdn_rho_max": MDN_RHO_MAX,
    }
