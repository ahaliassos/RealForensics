import torch
import torch.distributed as dist
import torch.nn as nn


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)


def copy_weights(net_from, net_to):
    for param_from, param_to in zip(net_from.parameters(), net_to.parameters()):
        param_to.data = param_from.data


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# ddp utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    tensors_gather = torch.cat(tensors_gather, dim=0)
    return tensors_gather


@torch.no_grad()
def concat_all_gather_var_len(tensor):
    """
    Performs all_gather operation on tensors with variable clips lengths.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # obtain Tensor number of frames of each rank
    world_size = dist.get_world_size()
    local_size = torch.LongTensor([tensor.shape[2]]).cuda()
    size_list = [torch.LongTensor([0]).cuda() for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    batch_size, channels, _, h, w = tensor.shape
    tensors_gather = [
        torch.ones(size=(batch_size, channels, max_size, h, w)).type_as(tensor) for _ in range(world_size)
    ]
    if local_size != max_size:
        padding = torch.zeros(size=(batch_size, channels, max_size - local_size, h, w)).cuda()
        tensor = torch.cat((tensor, padding), dim=2)
    dist.all_gather(tensors_gather, tensor)
    tensors_gather = torch.cat(tensors_gather, dim=0)
    return tensors_gather