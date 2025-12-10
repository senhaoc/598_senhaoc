import os
import random
import numpy as np
import torch

# Root of the Kaggle chest X-ray dataset
DATA_ROOT = os.environ.get("DATA_ROOT", "/content/data/pneumonia/chest_xray")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
