from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ..kan_conv.KANConv import KAN_Convolutional_Layer


class RKAN18(nn.Module):
    def __init__(self, num_classes = 1, version = "resnet18", kan_type = "chebyshev", pretrained = False, n_convs = 1, reduce_factor = [1, 2, 2, 2],
                 mechanisms = [None, None, None, "addition"], spline_order = (3, 2), grid_size = (3, 2), inv_bottleneck = False, inv_factor = 4, shortcut = False):
        super(RKAN18, self).__init__()

        self.mechanisms = mechanisms
        self.reduce_factor = reduce_factor
        self.inv_bottleneck = inv_bottleneck
        self.inv_factor = inv_factor
        self.shortcut = shortcut
        
        if pretrained:
            self.resnet = getattr(models, version)(weights = "DEFAULT")
        else:
            self.resnet = getattr(models, version)(weights = None)
        # self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)

        if len(self.mechanisms) != 4:
            raise ValueError(f"Length of mechanisms ({len(self.mechanisms)}) must match the number of stages (4).")

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        layer_config = {
            "resnet18": [64, 128, 256, 512],
            "resnet34": [64, 128, 256, 512],
            "resnet50": [256, 512, 1024, 2048],
            "resnet101": [256, 512, 1024, 2048],
            "resnet152": [256, 512, 1024, 2048]
        }
        channels = layer_config[version]

        # KAN convolutions for each stage
        self.kan_conv1 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1) if i == 0 else (2, 2), padding = (1, 1),
                                    kan_type = kan_type, spline_order = spline_order[0], grid_size = grid_size[0])
            for i in range(len(channels))
        ])

        self.kan_conv2 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1),
                                    kan_type = kan_type, spline_order = spline_order[1], grid_size = grid_size[1])
            for i in range(len(channels))
        ])

        # Bottleneck for KAN
        if self.inv_bottleneck:
            self.conv_reduce = nn.ModuleList([
                nn.Conv2d(64, 64 * self.inv_factor, kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[0], channels[0] * self.inv_factor, kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[1], channels[1] * self.inv_factor, kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[2], channels[2] * self.inv_factor, kernel_size = 1, stride = 1, bias = False)
            ])

            self.conv_expand = nn.ModuleList([
                nn.Conv2d((64 * self.inv_factor) * n_convs, channels[0], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d((channels[0] * self.inv_factor) * n_convs, channels[1], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d((channels[1] * self.inv_factor) * n_convs, channels[2], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d((channels[2] * self.inv_factor) * n_convs, channels[3], kernel_size = 1, stride = 1, bias = False)
            ])
        else:
            self.conv_reduce = nn.ModuleList([
                nn.Conv2d(64, 64 // reduce_factor[0], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[0], channels[0] // reduce_factor[1], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[1], channels[1] // reduce_factor[2], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[2], channels[2] // reduce_factor[3], kernel_size = 1, stride = 1, bias = False)
            ])

            self.conv_expand = nn.ModuleList([
                nn.Conv2d((64 // reduce_factor[0]) * n_convs, channels[0], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d((channels[0] // reduce_factor[1]) * n_convs, channels[1], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d((channels[1] // reduce_factor[2]) * n_convs, channels[2], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d((channels[2] // reduce_factor[3]) * n_convs, channels[3], kernel_size = 1, stride = 1, bias = False)
            ])

        if self.shortcut:     
            self.shortcut_bn = nn.ModuleList([nn.BatchNorm2d(ch) for ch in channels])
            self.conv_shortcut = nn.ModuleList([
                nn.Conv2d(64, channels[0], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[0], channels[1], kernel_size = 1, stride = 2, bias = False),
                nn.Conv2d(channels[1], channels[2], kernel_size = 1, stride = 2, bias = False),
                nn.Conv2d(channels[2], channels[3], kernel_size = 1, stride = 2, bias = False)
            ])

        # KAN normalization
        bn_params = {"momentum": 0.1, "eps": 1e-5, "affine": True}
        if self.inv_bottleneck:
            self.kan_bn = nn.ModuleList([nn.BatchNorm2d(ch * self.inv_factor, **bn_params) for i, ch in enumerate([64] + channels[:-1])])
        else:
            self.kan_bn = nn.ModuleList([nn.BatchNorm2d(ch // reduce_factor[i], **bn_params) for i, ch in enumerate([64] + channels[:-1])])
        self.kan_expand_bn = nn.ModuleList([nn.BatchNorm2d(ch, **bn_params) for ch in channels])

        # Activations
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()
        
        # Residual mechanisms
        # self.gate_convs = nn.ModuleList([nn.Conv2d(ch, ch, kernel_size = 1) for ch in channels])
        # self.se_blocks = nn.ModuleList([self._make_se_block(ch, reduction = 16) for ch in channels])
        self.name = "RKAN-18"
        # Head to project back to 1 channel for sequence output
        self.head_conv = nn.Conv2d(channels[-1], 1, kernel_size=1)
        self.out_conv = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)

    
    def apply_mechanism(self, out, residual, layer_index, mechanism):       
        if mechanism == "addition":
            return out + residual
        else:
            raise ValueError(f"Invalid mechanism: {mechanism}.")

    def forward(self, x):
        # x: (batch, 1, 2800) -> reshape to 2D by adding a height dim
        target_length = x.shape[-1]
        x = x.unsqueeze(1)  # (B,1,1,L)
        
        out = self.resnet.conv1(x)
        out = self.resnet.bn1(out)
        out = self.resnet.relu(out)
        out = self.resnet.maxpool(out)
        
        layers = [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]
        for i, (layer, mechanism) in enumerate(zip(layers, self.mechanisms)):
            identity = out
            out = layer(out)

            if mechanism is not None:
                residual = self.conv_reduce[i](identity)
                residual = self.silu(residual)
                residual = self.kan_conv1[i](residual)
                residual = self.kan_bn[i](residual)
                residual = self.conv_expand[i](residual)
                residual = self.silu(residual)
                
                if i == len(self.mechanisms) - 1:
                    residual = self.kan_conv2[i](residual)
                residual = self.kan_expand_bn[i](residual)

                if self.shortcut:
                    shortcut = self.conv_shortcut[i](identity)
                    shortcut = self.shortcut_bn[i](shortcut)
                    residual = residual + shortcut
                out = self.apply_mechanism(out, residual, i, mechanism)

        # out = self.head_conv(out)  # (B,1,H,W)
        out = self.resnet.avgpool(out.transpose(1, -1)).transpose(1, -1)
        out = F.interpolate(out, size=(1, target_length), mode="bilinear", align_corners=False)
        out = out.squeeze(2)  # (B,1,L)
        out = F.tanh(out)
        return out