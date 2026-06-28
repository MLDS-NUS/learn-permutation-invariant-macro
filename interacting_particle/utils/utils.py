import os
import random
import numpy as np
import torch


# Set the random seed for reproduction
def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

