import torch.nn as nn
import torch

noOfNeurons = 60
dilation = 1
kernel_size = 300
stride = 1
padding = int(((dilation*(kernel_size-1)-1)/stride-1)/2)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, noOfNeurons, kernel_size=kernel_size, stride=1, padding = padding+1),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv1d(noOfNeurons, 1, kernel_size=kernel_size, stride=1, padding = padding+2),
            nn.ReLU())
        self.cuda()
        self.name = 'CNN'
    
    def loss(self):
        return 0

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        return out

# model = CNNModel()

# print("Parameters count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))