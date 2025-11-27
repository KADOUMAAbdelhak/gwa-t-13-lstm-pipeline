import random
import numpy as np

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional dependency
    tf = None


def set_seed(seed: int):
    """
    Seed all relevant random number generators for reproducibility.
    TensorFlow seeding is best-effort and only applied when TF is installed.
    """
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)