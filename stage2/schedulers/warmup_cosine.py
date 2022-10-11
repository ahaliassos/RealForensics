import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    def __init__(
            self, optimizer, lr, warmup_epochs, num_epochs, iter_per_epoch, cosine_decay=True
    ):
        self.optimizer = optimizer
        self.base_lr = lr
        self.warmup_iter = int(warmup_epochs * iter_per_epoch)
        self.total_iter = int(num_epochs * iter_per_epoch)
        self.iter = 0
        self.current_lr = 0
        self.cosine_decay = cosine_decay

    def get_lr(self):
        if self.iter < self.warmup_iter:
            return self.base_lr * self.iter / self.warmup_iter
        elif not self.cosine_decay:
            return self.base_lr
        else:
            decay_iter = self.total_iter - self.warmup_iter
            return 0.5 * self.base_lr * (1 + np.cos(np.pi * (self.iter - self.warmup_iter) / decay_iter))

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_lr()
        self.iter += 1