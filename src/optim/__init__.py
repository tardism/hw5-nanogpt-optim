from .adam import Adam
from .rmsprop import RMSProp
from .signsgd import SignSGD

def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0, **kwargs):
    name = name.lower()
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
    if name == "rmsprop":
        return RMSProp(params, lr=lr, weight_decay=weight_decay, **kwargs)
    if name == "signsgd":
        return SignSGD(params, lr=lr, weight_decay=weight_decay, **kwargs)
    raise ValueError(f"Unknown optimizer: {name}")
