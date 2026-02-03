from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from ..kan import KANLinear


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class ResBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, *, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, dilation=dilation, stride=1, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm1d(planes)
        self.chomp1 = Chomp1d(padding)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, dilation=dilation, stride=1, padding=padding, bias=True)
        self.bn2 = nn.BatchNorm1d(planes)
        self.chomp2 = Chomp1d(padding)
        self.downsample = nn.Conv1d(inplanes, planes, 1, bias=True) if inplanes != planes else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.chomp1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.chomp2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResKAN(nn.Module):
    """
    ResKAN model used in the paper.
    Ported from `model/ResKAN.py::ResKAN` but avoids forced CUDA moves.
    """

    def __init__(
        self,
        res_layers: Sequence[int] = (1, 8, 8, 8, 16, 16, 16, 32, 32, 32),
        kan_layers: Sequence[int] = (3, 2, 1),
        *,
        grid_size: int = 5,
        kernel_size: int = 5,
    ):
        super().__init__()

        self.res_layers = nn.Sequential()
        self.res_layers2 = nn.Sequential()
        self.kan_layers = nn.Sequential()

        # Branch 1: hybrid dilations [1,2,5,...]
        for idx, (in_dim, out_dim) in enumerate(zip(res_layers, res_layers[1:])):
            dilation = [1, 2, 5][idx % 3]
            self.res_layers.append(ResBlock(in_dim, out_dim, kernel_size=kernel_size, dilation=dilation))
        self.res_layers.append(KANLinear(res_layers[-1], 1, grid_size=grid_size))

        # Branch 2: exponential dilations [1,2,4,8,...]
        for idx, (in_dim, out_dim) in enumerate(zip(res_layers, res_layers[1:])):
            dilation = 2**idx
            self.res_layers2.append(ResBlock(in_dim, out_dim, kernel_size=kernel_size, dilation=dilation))
        self.res_layers2.append(KANLinear(res_layers[-1], 1, grid_size=grid_size))

        for in_dim, out_dim in zip(kan_layers, kan_layers[1:]):
            self.kan_layers.append(KANLinear(in_dim, out_dim, grid_size=grid_size))

        self.name = "ResKAN"
        self.paper_name = "ResKAN"
        self.grid_size = grid_size
        self._arch_name = "ResKANet{}_[{}]_[{}]".format(
            grid_size, "x".join(map(str, res_layers)), "x".join(map(str, kan_layers))
        )

    @property
    def checkpoint_name(self) -> str:
        # Preserve the historical naming convention used by your existing checkpoints.
        return self._arch_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.res_layers(x)
        y2 = self.res_layers2(x)
        y = torch.cat((y, y2, x), dim=1)  # (B, 3, L)
        return self.kan_layers(y)  # (B, 1, L)


class ResKANLoss(nn.Module):
    """
    Default loss used in your scripts: weighted MSE (+ optional contour term if enabled).

    Notes:
    - To keep backward compatibility with your current training setup, CE term is optional.
    - When loss_weight[1] == 0, this reduces to plain MSE.
    """

    def __init__(self, loss_weight: Sequence[float] = (1.0, 0.0)) -> None:
        super().__init__()
        self.loss_weight = list(loss_weight)
        self.mse = nn.MSELoss()

        # Keep the same class-weight convention as your original code (2 classes)
        ew = torch.from_numpy(np.array([1, 1], dtype=np.float32))
        self.register_buffer("class_weight", ew)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weight)

    def forward(self, out: torch.Tensor, gt: torch.Tensor, co: torch.Tensor, epoch: int) -> torch.Tensor:
        loss1 = self.mse(out, gt)

        # Optional contour-aware CE (paper describes using logits [y', 1-y'])
        loss2 = torch.tensor(0.0, device=out.device)
        if len(self.loss_weight) > 1 and self.loss_weight[1] and self.loss_weight[1] > 0:
            # logits: (B, 2, L)
            logits = torch.cat([out, 1 - out], dim=1)
            logits = logits.permute(0, 2, 1).reshape(-1, 2)
            labels = co.squeeze()
            if labels.dim() == 2:
                labels = labels.reshape(-1)
            labels = labels.long()
            loss2 = self.ce(logits, labels)

        return self.loss_weight[0] * loss1 + (self.loss_weight[1] * loss2 if len(self.loss_weight) > 1 else 0.0)

