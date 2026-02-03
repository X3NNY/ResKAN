from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, bias=True) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_channels: list[int], kernel_size: int = 2, dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNModel(nn.Module):
    """
    TCN baseline (ported from `model/TCN.py`).
    Input/Output: (B, 1, L) -> (B, 1, L)
    """

    def __init__(self, input_size: int, output_size: int, num_channels: list[int], kernel_size: int, dropout: float):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.global_residual = nn.Conv1d(input_size, output_size, kernel_size=1)  # Global skip from input to last layer
        self.out = nn.Linear(num_channels[-1]+1, output_size)
        self.weight_init_res()
        self.name = "TCN"

    def weight_init_res(self):
        self.global_residual.weight.data.normal_(0, 0.01)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        y = self.tcn(inputs)
        y = torch.cat((y, inputs), dim=1)
        return self.out(y.transpose(1, 2)).transpose(1, 2)

