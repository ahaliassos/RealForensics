import random

import torch
import torch.nn as nn


class TimeMask:
    """time mask for 1d specAug"""

    def __init__(self, T=8, n_mask=1, replace_with_zero=False):
        self.n_mask = n_mask
        self.T = T
        self.replace_with_zero = replace_with_zero

    def __call__(self, x, idxs=None):
        cloned = x.clone()
        len_raw = cloned.size(1)

        if idxs is None:
            idxs = []
            for i in range(self.n_mask):
                t = random.randrange(self.T)
                if len_raw - t <= 0:
                    continue
                t_zero = random.randrange(0, len_raw - t)
                # avoids randrange error if values are equal and range is empty
                if t_zero == t_zero + t:
                    continue
                mask_end = t_zero + t
                idxs.extend(list(range(t_zero, mask_end)))

        if self.replace_with_zero:
            cloned[:, idxs] = 0
        else:
            cloned[:, idxs] = cloned.mean()
        return cloned, idxs


class TimeMaskAudio:
    """time mask for 1d specAug"""

    def __init__(self, T=8, n_mask=1, downsample=4, replace_with_zero=False):
        self.n_mask = n_mask
        self.T = T
        self.downsample = downsample  # Only used when idxs is supplied in __call__
        self.replace_with_zero = replace_with_zero

    def __call__(self, x, idxs=None):
        cloned = x.clone()
        len_raw = cloned.size(1)

        if idxs is None:
            idxs = []
            for i in range(self.n_mask):
                t = random.randrange(self.T)
                if len_raw - t <= 0:
                    continue
                t_zero = random.randrange(0, len_raw - t)
                # avoids randrange error if values are equal and range is empty
                if t_zero == t_zero + t:
                    continue
                mask_end = t_zero + t
                idxs.extend(list(range(t_zero, mask_end)))
        else:
            idxs = [idx * self.downsample + j for idx in idxs for j in range(self.downsample)]

        if self.replace_with_zero:
            cloned[:, idxs] = 0
        else:
            cloned[:, idxs] = cloned.mean()
        return cloned, idxs


class TimeMaskV2:
    def __init__(self, p=0.5, T=8, replace_with_zero=False):
        self.p = p
        self.T = T
        self.replace_with_zero = replace_with_zero

    def __call__(self, x, idxs=None):
        if idxs is None and torch.rand(1) < self.p:
            return x, []

        cloned = x.clone()

        if idxs is None:
            len_raw = cloned.size(1)
            num_to_zero = random.randrange(1, self.T + 1)
            idxs = random.sample(range(len_raw), num_to_zero)

        if self.replace_with_zero:
            cloned[:, idxs] = 0
        else:
            cloned[:, idxs] = cloned.mean()
        return cloned, idxs


class TimeMaskAudioV2:
    def __init__(self, p=0.5, T=8, downsample=4, replace_with_zero=False):
        self.p = p
        self.T = T
        self.downsample = downsample
        self.replace_with_zero = replace_with_zero

    def __call__(self, x, idxs=None):
        if idxs is None and torch.rand(1) < self.p:
            return x, []

        cloned = x.clone()

        if idxs is None:
            len_raw = cloned.size(1) // self.downsample
            num_to_zero = random.randrange(1, self.T // self.downsample + 1)
            idxs = random.sample(range(len_raw), num_to_zero)

        idxs = [idx * self.downsample + j for idx in idxs for j in range(self.downsample)]
        if self.replace_with_zero:
            cloned[:, idxs] = 0
        else:
            cloned[:, idxs] = cloned.mean()
        return cloned, idxs


class FrequencyMask:
    """time mask for 1d specAug"""

    def __init__(self, F=8, n_mask=1, replace_with_zero=False):
        self.n_mask = n_mask
        self.F = F
        self.replace_with_zero = replace_with_zero

    def __call__(self, x):
        cloned = x.clone()
        len_raw = cloned.size(2)
        for i in range(self.n_mask):
            f = random.randrange(self.F)
            if len_raw - f <= 0:
                continue
            f_zero = random.randrange(0, len_raw - f)
            # avoids randrange error if values are equal and range is empty
            if f_zero == f_zero + f:
                continue
            mask_end = f_zero + f
            if self.replace_with_zero:
                cloned[:, :, f_zero:mask_end] = 0
            else:
                cloned[:, :, f_zero:mask_end] = cloned.mean()
        return cloned


class FrequencyMaskV2:
    """time mask for 1d specAug"""

    def __init__(self, p=0.5, F=8, downsample=4, replace_with_zero=False):
        self.p = p
        self.F = F
        self.downsample = downsample
        self.replace_with_zero = replace_with_zero

    def __call__(self, x):
        if torch.rand(1) < self.p:
            cloned = x.clone()
            len_raw = cloned.size(2) // self.downsample
            num_to_zero = random.randrange(1, self.F // self.downsample + 1)
            idxs = random.sample(range(len_raw), num_to_zero)
            idxs = [idx * self.downsample + j for idx in idxs for j in range(self.downsample)]
            if self.replace_with_zero:
                cloned[:, :, idxs] = 0
            else:
                cloned[:, :, idxs] = cloned.mean()
            return cloned
        return x


class ZeroPadTemp:
    """time mask for 1d specAug"""

    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        ones = torch.ones(x.size(1), dtype=torch.long, device=x.device)
        zeros = torch.zeros((self.size - x.size(1)), dtype=torch.long, device=x.device)
        if x.size(1) < self.size:
            new = torch.zeros((x.size(0), self.size, *x.shape[2:]), dtype=x.dtype, device=x.device)
            new[:, :x.size(1), ...] = x
            x = new
        return x, torch.cat([ones, zeros])


class LambdaModule(nn.Module):
    def __init__(self, lambda_fn):
        super().__init__()
        self.lambda_fn = lambda_fn

    def forward(self, x):
        return self.lambda_fn(x)