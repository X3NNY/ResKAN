import torch.nn as nn
import torch
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 4
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.inchannel),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 4, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 8, 2, stride=1)
        self.layer3 = self.make_layer(ResBlock, 16, 2, stride=1)        
        self.layer4 = self.make_layer(ResBlock, 32, 2, stride=1)        
        self.fc = nn.Linear(32, num_classes)
        self.name = 'ResNet'
        self.cuda()
    
    def loss(self):
        return 0

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        out = self.fc(out.transpose(-1, -2)).transpose(-1, -2)
        return out

# model = ResNet(BasicBlock, 1)

# print("Parameters count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))