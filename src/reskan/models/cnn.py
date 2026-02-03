from __future__ import annotations

import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """
    CNN baseline used in this project (ported from `model/CNN.py`).
    Input/Output: (B, 1, L) -> (B, 1, L)
    """

    def __init__(
        self,
        *,
        no_of_neurons: int = 60,
        kernel_size: int = 300,
        dilation: int = 1,
    ):
        super().__init__()
        padding = int(((dilation * (kernel_size - 1) - 1) / 1 - 1) / 2)

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, no_of_neurons, kernel_size=kernel_size, stride=1, padding=padding + 1),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(no_of_neurons, 1, kernel_size=kernel_size, stride=1, padding=padding + 2),
            nn.Tanh(),
        )

        self.name = "CNN"
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(x)
        return self.layer2(out)

