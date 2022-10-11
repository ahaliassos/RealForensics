import torch.nn as nn


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
