import torch

class RMSProp:
    """Minimal RMSProp with optional momentum (defaults to 0).

    Update:
      v <- alpha * v + (1-alpha) * g^2
      p <- p - lr * g / (sqrt(v) + eps)

    If momentum > 0:
      m <- momentum * m + lr * g / (sqrt(v)+eps)
      p <- p - m

    Args:
      alpha: running average coefficient (PyTorch RMSprop uses alpha=0.99 by default)
    """
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8, momentum=0.0, weight_decay=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = [torch.zeros_like(p) for p in self.params]
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

            self.v[i].mul_(self.alpha).addcmul_(g, g, value=1 - self.alpha)
            denom = self.v[i].sqrt().add_(self.eps)

            if self.momentum and self.m is not None:
                self.m[i].mul_(self.momentum).addcdiv_(g, denom, value=self.lr)
                p.add_(self.m[i], alpha=-1.0)
            else:
                p.addcdiv_(g, denom, value=-self.lr)
