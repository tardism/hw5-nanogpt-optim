import os, random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism helps comparability but can slow training slightly.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
