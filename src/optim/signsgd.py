import torch

class SignSGD:
    """SignSGD (optionally with momentum on the sign updates).

    Update:
      u <- sign(g)
      (optional) m <- mu*m + u
      p <- p - lr * u (or lr*m)

    This optimizer is simple and can be surprisingly robust, but may need larger lrs.

    Args:
      momentum: momentum coefficient on sign updates (0 disables)
      weight_decay: L2 penalty (applied to gradient before sign)
    """
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params] if momentum > 0 else None

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0.0:
                g = g.add(p, alpha=self.weight_decay)

            u = g.sign()

            if self.momentum and self.m is not None:
                self.m[i].mul_(self.momentum).add_(u)
                p.add_(self.m[i].sign(), alpha=-self.lr)
            else:
                p.add_(u, alpha=-self.lr)
