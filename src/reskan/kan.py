"""
KAN implementation used by ResKAN (adapted from `model/KAN.py`).

Key adaptation for this cleaned repo:
- Do NOT automatically move modules to CUDA in `__init__`.
- Keep forward signature compatible with your existing ResKAN usage.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list[float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise)
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)  # (in, batch, coeff)
        B = y.transpose(0, 1)  # (in, batch, out)
        solution = torch.linalg.lstsq(A, B).solution  # (in, coeff, out)
        result = solution.permute(2, 0, 1)  # (out, in, coeff)

        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expected input: (B, C_in, L). Internally treated as per-time-step features.
        Output: (B, C_out, L)
        """
        x = x.transpose(1, 2)  # (B, L, C)
        assert x.dim() == 3 and x.size(2) == self.in_features

        def f(x2: torch.Tensor) -> torch.Tensor:
            base_output = F.linear(self.base_activation(x2), self.base_weight)
            spline_output = F.linear(
                self.b_splines(x2).view(x2.size(0), -1),
                self.scaled_spline_weight.view(self.out_features, -1),
            )
            return base_output + spline_output

        out = torch.stack([f(x[i]) for i in range(x.shape[0])], dim=0)  # (B, L, out)
        return out.transpose(1, 2)  # (B, out, L)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0) -> torch.Tensor:
        l1_fake = self.spline_weight.abs().mean(-1)
        reg_act = l1_fake.sum()
        p = l1_fake / reg_act
        reg_ent = -torch.sum(p * p.log())
        return regularize_activation * reg_act + regularize_entropy * reg_ent


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden: list[int],
        *,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list[float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=list(grid_range),
                )
            )

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        if update_grid:
            raise NotImplementedError("Grid update is not exposed in this cleaned wrapper.")
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0) -> torch.Tensor:
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

