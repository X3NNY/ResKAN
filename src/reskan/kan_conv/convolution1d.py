# Credits to: https://github.com/detkov/Convolution-From-Scratch/
# Adapted to 1D convolution for KAN
import numpy as np
import torch
from typing import List


def calc_out_dims_1d(matrix, kernel_size, stride, dilation, padding):
    batch_size, n_channels, length = matrix.shape
    l_out = (
        np.floor((length + 2 * padding - kernel_size - (kernel_size - 1) * (dilation - 1)) / stride).astype(int)
        + 1
    )
    return l_out, batch_size, n_channels


def kan_conv1d(
    matrix: torch.Tensor,
    kernel: torch.nn.Module,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 0,
    device: str = "cuda",
) -> torch.Tensor:
    l_out, batch_size, n_channels = calc_out_dims_1d(matrix, kernel_size, stride, dilation, padding)

    matrix_2d = matrix.unsqueeze(2)  # (B, C, 1, L)
    unfold = torch.nn.Unfold(
        kernel_size=(1, kernel_size),
        dilation=(1, dilation),
        padding=(0, padding),
        stride=(1, stride),
    )
    unfolded_matrix = unfold(matrix_2d)  # (B, C*kernel, l_out)
    unfolded_matrix = unfolded_matrix.view(batch_size, n_channels, kernel_size, l_out)
    conv_groups = unfolded_matrix.permute(0, 1, 3, 2)  # (B, C, l_out, kernel)

    conv_groups_flat = conv_groups.reshape(-1, kernel_size)
    conv_result = kernel(conv_groups_flat)  # (B*C*l_out, 1)
    return conv_result.view(batch_size, n_channels, l_out)


def multiple_convs_kan_conv1d(
    matrix: torch.Tensor,
    kernels: List[torch.nn.Module],
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 0,
    device: str = "cuda",
) -> torch.Tensor:
    l_out, batch_size, n_channels = calc_out_dims_1d(matrix, kernel_size, stride, dilation, padding)

    matrix_2d = matrix.unsqueeze(2)
    unfold = torch.nn.Unfold(
        kernel_size=(1, kernel_size),
        dilation=(1, dilation),
        padding=(0, padding),
        stride=(1, stride),
    )
    unfolded_matrix = unfold(matrix_2d)
    unfolded_matrix = unfolded_matrix.view(batch_size, n_channels, kernel_size, l_out)
    conv_groups = unfolded_matrix.permute(0, 1, 3, 2)

    conv_groups_flat = conv_groups.reshape(-1, kernel_size)

    results = []
    for kernel in kernels:
        conv_result = kernel.conv.forward(conv_groups_flat)
        results.append(conv_result.view(batch_size, n_channels, l_out))

    return torch.cat(results, dim=1)

