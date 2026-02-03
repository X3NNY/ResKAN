from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import conv1d


class RNNInverseModel(nn.Module):
    """
    RNN baseline (renamed from EIIN inversion network).
    Ported from `model/EIIN.py::inverse_model`.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        resolution_ratio: int = 6,
        nonlinearity: str = "tanh",
        factor: float = 3.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.resolution_ratio = resolution_ratio
        self.activation = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()

        # NOTE: original code uses num_groups=self.in_channels (typically 1)
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=round(8 * factor), kernel_size=5, padding=2, dilation=1),
            nn.GroupNorm(num_groups=self.in_channels, num_channels=round(8 * factor)),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=round(8 * factor), kernel_size=5, padding=6, dilation=3),
            nn.GroupNorm(num_groups=self.in_channels, num_channels=round(8 * factor)),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=round(8 * factor), kernel_size=5, padding=12, dilation=6),
            nn.GroupNorm(num_groups=self.in_channels, num_channels=round(8 * factor)),
        )

        self.cnn = nn.Sequential(
            self.activation,
            nn.Conv1d(in_channels=round(24 * factor), out_channels=int(16 * factor), kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=self.in_channels, num_channels=round(16 * factor)),
            self.activation,
            nn.Conv1d(in_channels=round(16 * factor), out_channels=round(16 * factor), kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=self.in_channels, num_channels=round(16 * factor)),
            self.activation,
            nn.Conv1d(in_channels=round(16 * factor), out_channels=round(16 * factor), kernel_size=1),
            nn.GroupNorm(num_groups=self.in_channels, num_channels=round(16 * factor)),
            self.activation,
        )

        self.gru = nn.GRU(
            input_size=self.in_channels, hidden_size=round(8 * factor), num_layers=3, batch_first=True, bidirectional=True
        )

        self.up = nn.Sequential(
            nn.ConvTranspose1d(in_channels=round(16 * factor), out_channels=round(8 * factor), stride=3, kernel_size=5, padding=1),
            nn.GroupNorm(num_groups=self.in_channels, num_channels=round(8 * factor)),
            self.activation,
            nn.ConvTranspose1d(in_channels=round(8 * factor), out_channels=round(8 * factor), stride=2, kernel_size=4, padding=1),
            nn.GroupNorm(num_groups=self.in_channels, num_channels=round(8 * factor)),
            self.activation,
        )

        self.gru_out = nn.GRU(input_size=round(8 * factor), hidden_size=round(8 * factor), num_layers=1, batch_first=True, bidirectional=True)
        self.out = nn.Linear(in_features=round(16 * factor), out_features=self.in_channels)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.name = "RNN"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_out1 = self.cnn1(x)
        cnn_out2 = self.cnn2(x)
        cnn_out3 = self.cnn3(x)
        cnn_out = self.cnn(torch.cat((cnn_out1, cnn_out2, cnn_out3), dim=1))

        tmp_x = x.transpose(-1, -2)  # (B, L, C)
        rnn_out, _ = self.gru(tmp_x)
        rnn_out = rnn_out.transpose(-1, -2)  # (B, 2H, L)

        x2 = rnn_out + cnn_out
        x2 = self.up(x2)

        tmp_x = x2.transpose(-1, -2)
        x2, _ = self.gru_out(tmp_x)

        x2 = self.out(x2)
        x2 = x2.transpose(-1, -2)[..., :: self.resolution_ratio]
        return x2


