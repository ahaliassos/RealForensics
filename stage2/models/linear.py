import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage2.models.utils import Lambda


class MeanLinear(nn.Module):
    def __init__(self, in_dim, out_dim=1, norm_linear=False, scale=64):
        super().__init__()

        if norm_linear:
            self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.scale = scale
            self.linear = Lambda(lambda x: F.linear(F.normalize(x), F.normalize(self.weight)) * self.scale)
        else:
            self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = x.mean(-1)
        return self.linear(x)