import torch.nn as nn
import torch
from model.KAN import KANLinear

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, kernel_size=7,dilation=1):
        super(ResBlock, self).__init__()
        padding = (kernel_size-1) * dilation
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm1d(planes)
        self.chomp1 = Chomp1d(padding)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, bias=True)
        self.bn2 = nn.BatchNorm1d(planes)
        self.chomp2 = Chomp1d(padding)
        self.downsample = nn.Conv1d(inplanes, planes, 1, bias=True) if inplanes != planes else None
        self.stride = stride

        self.init_weights()
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)


    def forward(self, x):
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

        out += residual
        return self.relu(out)

class ResKAN(nn.Module):
    name = None

    def __init__(self, res_layers, kan_layers, *,
                grid_size: int = 5,
                kernel_size: int = 7,
                device: str = 'cuda') -> None:
        super(ResKAN, self).__init__()

        self.res_layers = nn.Sequential()
        self.res_layers2 = nn.Sequential()
        # self.res_layers3 = nn.Sequential()
        self.kan_layers = nn.ModuleList()
        # self.kan_layers = KAN(kan_layers, 5, device='cuda')

        for idx, (in_dim, out_dim) in enumerate(zip(res_layers, res_layers[1:])):
            dilation = 2 ** idx
            self.res_layers.append(
                ResBlock(
                    in_dim, out_dim,
                    kernel_size=kernel_size,
                    dilation=dilation
                )
            )
        
        for idx, (in_dim, out_dim) in enumerate(zip(res_layers, res_layers[1:])):
            dilation = 2 ** idx
            self.res_layers2.append(
                ResBlock(
                    in_dim, out_dim,
                    kernel_size=kernel_size,
                    dilation=dilation
                )
            )
        

        for in_dim, out_dim in zip(kan_layers, kan_layers[1:]):
            self.kan_layers.append(
                KANLinear(
                    in_dim, out_dim,
                    grid_size=grid_size
                )
            )
        
        self.name = 'ResKANet{}_[{}]_[{}]'.format(grid_size, 'x'.join(map(str, res_layers)),
                                                'x'.join(map(str, kan_layers)))

        self.to(device=device)

    def loss(self):
        # def reg(acts_scale):
        #     def nonlinear(x, th=1e-16, factor=1.):
        #         return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)
        #     reg_ = 0.
        #     for i in range(len(acts_scale)):
        #         vec = acts_scale[i].reshape(-1, )

        #         p = vec / torch.sum(vec)
        #         l1 = torch.sum(nonlinear(vec))
        #         entropy = - torch.sum(p * torch.log2(p + 1e-4))
        #         reg_ += 5 * l1 + 10 * entropy  # both l1 and entropy

        #     # regularize coefficient to encourage spline to be zero
        #     # for i in range(len(self.act_fun)):
        #     #     coeff_l1 = torch.sum(torch.mean(torch.abs(self.act_fun[i].coef), dim=1))
        #     #     coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.act_fun[i].coef)), dim=1))
        #     #     reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

        #     return reg_
        # return reg(self.kan_layers.acts_scale)
        return sum(layer.regularization_loss(1, 1) for layer in self.kan_layers)
    
    def forward(self, x: torch.Tensor, all=False):
        y = self.res_layers(x)
        y2 = self.res_layers2(x)
        y = torch.cat((y, y2, x), dim=1)

        out = y.transpose(1,2)
        for layer in self.kan_layers:
            out = torch.stack([layer(out[i]) for i in range(out.shape[0])], dim=0)
        out = out.transpose(1, 2)

        # out = y.transpose(1,2)
        # out = torch.stack([self.kan_layers(out[i]) for i in range(out.shape[0])], dim=0)
        # out = out.transpose(1, 2)
        return out


class ResFCN(nn.Module):
    name = None

    def __init__(self, res_layers, fc_layers, *,
                kernel_size: int = 7,
                device: str = 'cuda') -> None:
        super(ResFCN, self).__init__()

        self.res_layers = nn.Sequential()
        self.res_layers2 = nn.Sequential()
        self.fc_layers = nn.Sequential()

        for idx, (in_dim, out_dim) in enumerate(zip(res_layers, res_layers[1:])):
            dilation = 2 ** idx
            self.res_layers.append(
                ResBlock(
                    in_dim, out_dim,
                    kernel_size=kernel_size,
                    dilation=dilation
                )
            )
        
        for idx, (in_dim, out_dim) in enumerate(zip(res_layers, res_layers[1:])):
            dilation = 2 ** idx
            self.res_layers2.append(
                ResBlock(
                    in_dim, out_dim,
                    kernel_size=kernel_size,
                    dilation=dilation
                )
            )

        self.fc = nn.Linear(res_layers[-1]+2, 1)
        self.fc.weight.data.normal_(0, 0.01)
        
        self.name = 'ResFCNet_[{}]'.format('x'.join(map(str, res_layers)))

        self.to(device=device)
    
    def loss(self):
        return 0
    
    def forward(self, x: torch.Tensor):
        y = self.res_layers(x)
        y2 = self.res_layers2(x)
        y = torch.cat((y, y2, x), dim=1)
        out = self.fc(y.transpose(1, 2)).transpose(1, 2)
        # out = self.fc_layers(y.transpose(1, 2)).transpose(1, 2)
        return out


class ResMLPN(nn.Module):
    name = None

    def __init__(self, res_layers, fc_layers, *,
                kernel_size: int = 7,
                device: str = 'cuda') -> None:
        super(ResMLPN, self).__init__()

        self.res_layers = nn.Sequential()
        self.res_layers2 = nn.Sequential()
        self.fc_layers = nn.Sequential()

        for idx, (in_dim, out_dim) in enumerate(zip(res_layers, res_layers[1:])):
            dilation = 2 ** idx
            self.res_layers.append(
                ResBlock(
                    in_dim, out_dim,
                    kernel_size=kernel_size,
                    dilation=dilation
                )
            )

        
        for idx, (in_dim, out_dim) in enumerate(zip(res_layers, res_layers[1:])):
            dilation = 2 ** idx
            self.res_layers2.append(
                ResBlock(
                    in_dim, out_dim,
                    kernel_size=kernel_size,
                    dilation=dilation
                )
            )

        for idx, (in_dim, out_dim) in enumerate(zip(fc_layers, fc_layers[1:])):
            if idx != 0:
                self.fc_layers.append(nn.ReLU(inplace=True))
            self.fc_layers.append(
                nn.Linear(in_dim, out_dim)
            )
            self.fc_layers[-1].weight.data.normal_(0, 0.01)
        
        self.name = 'ResMLPNet_[{}]_[{}]'.format('x'.join(map(str, res_layers)),
                                                'x'.join(map(str, fc_layers)))

        self.to(device=device)
    
    def loss(self):
        return 0
    
    def forward(self, x: torch.Tensor):
        y = self.res_layers(x)
        y2 = self.res_layers2(x)
        y = torch.cat((y, y2, x), dim=1)
        out = self.fc_layers(y.transpose(1, 2)).transpose(1, 2)
        return out
    

class ResCN(nn.Module):
    name = None

    def __init__(self, res_layers, fc_layers, *,
                kernel_size: int = 7,
                device: str = 'cuda') -> None:
        super(ResCN, self).__init__()

        self.res_layers = nn.Sequential()
        self.res_layers2 = nn.Sequential()
        self.fc_layers = nn.Sequential()

        for idx, (in_dim, out_dim) in enumerate(zip(res_layers, res_layers[1:])):
            dilation = 2 ** idx
            self.res_layers.append(
                ResBlock(
                    in_dim, out_dim,
                    kernel_size=kernel_size,
                    dilation=dilation
                )
            )
        
        for idx, (in_dim, out_dim) in enumerate(zip(res_layers, res_layers[1:])):
            dilation = 2 ** idx
            self.res_layers2.append(
                ResBlock(
                    in_dim, out_dim,
                    kernel_size=kernel_size,
                    dilation=dilation
                )
            )

        self.fc = nn.Conv1d(3, 1, kernel_size=1)
        self.fc.weight.data.normal_(0, 0.01)
        
        self.name = 'ResCNet_[{}]'.format('x'.join(map(str, res_layers)))

        self.to(device=device)
    
    def loss(self):
        return 0
    
    def forward(self, x: torch.Tensor):
        y = self.res_layers(x)
        y2 = self.res_layers2(x)
        y = torch.cat((y, y2, x), dim=1)
        return self.fc(y)