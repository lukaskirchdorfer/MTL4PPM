import random, torch
import numpy as np

def set_random_seed(seed):
    r"""Set the random seed for reproducibility.

    Args:
        seed (int, default=42): The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)             # For single GPU
        torch.cuda.manual_seed_all(seed)         # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False