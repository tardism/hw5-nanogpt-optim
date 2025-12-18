import os
import numpy as np
import torch
from typing import Tuple

def _load_bin(path: str) -> np.memmap:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Expected pre-tokenized memmap binaries. "
            f"Create data/train.bin and data/val.bin or change --data_dir."
        )
    # nanoGPT convention: uint16 token ids
    return np.memmap(path, dtype=np.uint16, mode="r")

def get_batch(split: str, data_dir: str, batch_size: int, block_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_path = os.path.join(data_dir, f"{split}.bin")
    data = _load_bin(bin_path)
    # random starting indices
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y
