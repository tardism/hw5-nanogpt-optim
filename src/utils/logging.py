import csv, json, os
from dataclasses import asdict
from typing import Dict, Any, Optional

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

class MetricsWriter:
    def __init__(self, out_path: str):
        self.out_path = out_path
        self._f = open(out_path, "w", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=["step","train_loss","val_loss","val_ppl","lr"])
        self._writer.writeheader()

    def write(self, step: int, train_loss: float, val_loss: float, val_ppl: float, lr: float) -> None:
        self._writer.writerow({
            "step": step,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_ppl": float(val_ppl),
            "lr": float(lr),
        })
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
