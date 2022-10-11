# adapted from https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/csn.py

from typing import Callable, Tuple

import torch
import torch.nn as nn
from pytorchvideo.models.resnet import Net, create_bottleneck_block, create_res_stage
from pytorchvideo.models.stem import create_res_basic_stem


class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def create_csn(
    *,
    # Input clip configs.
    input_channel: int = 3,
    # Model configs.
    model_depth: int = 50,
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem configs.
    stem_dim_out: int = 64,
    stem_conv_kernel_size: Tuple[int] = (3, 7, 7),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    stem_pool: Callable = None,
    stem_pool_kernel_size: Tuple[int] = (1, 3, 3),
    stem_pool_stride: Tuple[int] = (1, 2, 2),
    # Stage configs.
    stage_conv_a_kernel_size: Tuple[int] = (1, 1, 1),
    stage_conv_b_kernel_size: Tuple[int] = (3, 3, 3),
    stage_conv_b_width_per_group: int = 1,
    stage_spatial_stride: Tuple[int] = (1, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 2, 2, 2),
    bottleneck: Callable = create_bottleneck_block,
    bottleneck_ratio: int = 4,
) -> nn.Module:

    torch._C._log_api_usage_once("PYTORCHVIDEO.model.create_csn")

    # Number of blocks for different stages given the model depth.
    _MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}

    # Given a model depth, get the number of blocks for each stage.
    assert (
        model_depth in _MODEL_STAGE_DEPTH.keys()
    ), f"{model_depth} is not in {_MODEL_STAGE_DEPTH.keys()}"
    stage_depths = _MODEL_STAGE_DEPTH[model_depth]

    blocks = []
    # Create stem for CSN.
    stem = create_res_basic_stem(
        in_channels=input_channel,
        out_channels=stem_dim_out,
        conv_kernel_size=stem_conv_kernel_size,
        conv_stride=stem_conv_stride,
        conv_padding=[size // 2 for size in stem_conv_kernel_size],
        pool=stem_pool,
        pool_kernel_size=stem_pool_kernel_size,
        pool_stride=stem_pool_stride,
        pool_padding=[size // 2 for size in stem_pool_kernel_size],
        norm=norm,
        activation=activation,
    )
    blocks.append(stem)

    stage_dim_in = stem_dim_out
    stage_dim_out = stage_dim_in * 4

    # Create each stage for CSN.
    for idx in range(len(stage_depths)):
        stage_dim_inner = stage_dim_out // bottleneck_ratio
        depth = stage_depths[idx]

        stage_conv_b_stride = (
            stage_temporal_stride[idx],
            stage_spatial_stride[idx],
            stage_spatial_stride[idx],
        )

        stage = create_res_stage(
            depth=depth,
            dim_in=stage_dim_in,
            dim_inner=stage_dim_inner,
            dim_out=stage_dim_out,
            bottleneck=bottleneck,
            conv_a_kernel_size=stage_conv_a_kernel_size,
            conv_a_stride=(1, 1, 1),
            conv_a_padding=[size // 2 for size in stage_conv_a_kernel_size],
            conv_b_kernel_size=stage_conv_b_kernel_size,
            conv_b_stride=stage_conv_b_stride,
            conv_b_padding=[size // 2 for size in stage_conv_b_kernel_size],
            conv_b_num_groups=(stage_dim_inner // stage_conv_b_width_per_group),
            conv_b_dilation=(1, 1, 1),
            norm=norm,
            activation=activation,
        )

        blocks.append(stage)
        stage_dim_in = stage_dim_out
        stage_dim_out = stage_dim_out * 2

    blocks.append(LambdaModule(lambda x: x.mean(dim=(3, 4))))

    return Net(blocks=nn.ModuleList(blocks))


def csn_temporal_no_head(**kwargs):
    model = create_csn(stem_pool=nn.MaxPool3d, stage_temporal_stride=(1, 1, 1, 1), **kwargs)
    return model
