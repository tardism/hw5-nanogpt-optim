import math
import torch

class Adam:
    """Minimal AdamW-style optimizer (decoupled weight decay).
    This mirrors the style often used in course notebooks: a lightweight class
    with `step()` and `zero_grad()`.

    Args:
        params: iterable of parameters
        lr: learning rate
        beta1, beta2: moment decay factors
        eps: numerical stability
        weight_decay: decoupled weight decay coefficient
    """
    def __init__(self, params, lr=3e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        lr = self.lr

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad

            # decoupled weight decay (AdamW)
            if self.weight_decay != 0.0:
                p.add_(p, alpha=-lr * self.weight_decay)

            self.m[i].mul_(b1).add_(g, alpha=1 - b1)
            self.v[i].mul_(b2).addcmul_(g, g, value=1 - b2)

            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)

            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-lr)
