import torch
import torch.nn as nn
from quantization import *
from config import SearchConfig

args = SearchConfig()

OPS = {
    'none' : lambda C, stride, affine: Zero(stride),
    'max_pool_3x3_1.25' : lambda C, stride, affine: MaxPool(3, stride=stride, padding=1, base=1.25, time_step=args.timestep),
    'max_pool_3x3_1.5' : lambda C, stride, affine: MaxPool(3, stride=stride, padding=1, base=1.5, time_step=args.timestep),
    'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'conv_3x3_1.25' : lambda C, stride, affine: Conv(C, C, 3, stride, 1, base=1.25, time_step=args.timestep, affine=affine),
    'conv_3x3_1.5' : lambda C, stride, affine: Conv(C, C, 3, stride, 1, base=1.5, time_step=args.timestep, affine=affine),
    'sep_conv_3x3_1.25' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, base=1.25, time_step=args.timestep, affine=affine),
    'sep_conv_3x3_1.5' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, base=1.5, time_step=args.timestep, affine=affine),
    'dil_conv_3x3_1.25' : lambda C, stride, affine: DilConv(C, C, 3, stride, 1, 2, base=1.25, time_step=args.timestep, affine=affine),
    'dil_conv_3x3_1.5' : lambda C, stride, affine: DilConv(C, C, 3, stride, 1, 2, base=1.5, time_step=args.timestep, affine=affine)
}

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            PACT()
        )

    def forward(self, x):
        return self.op(x)

class AvgPool(nn.Module):
    def __init__(self, kernel_size, stride, padding, base, time_step):
        super(AvgPool, self).__init__()
        self.op = nn.Sequential(
        nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=False),
        PACT_with_log_quantize(base, time_step)
        )
    
    def forward(self, x):
        return self.op(x)

class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding, base, time_step):
        super(MaxPool, self).__init__()
        self.op = nn.Sequential(
        nn.MaxPool2d(kernel_size, stride=stride, padding=padding),
        PACT_with_log_quantize(base, time_step)
        )
    
    def forward(self, x):
        return self.op(x)

class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, base, time_step, affine=True):
        super(Conv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            PACT_with_log_quantize(base, time_step)
            )
    
    def forward(self, x):
        return self.op(x)
    
class DilConv(nn.Module):
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, base, time_step, affine=True):
        super(DilConv, self).__init__()    
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            PACT_with_log_quantize(base, time_step),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            PACT_with_log_quantize(base, time_step)
            )

    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, base, time_step, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            PACT_with_log_quantize(base, time_step),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            PACT_with_log_quantize(base, time_step),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            PACT_with_log_quantize(base, time_step),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            PACT_with_log_quantize(base, time_step)
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, base, time_step, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(C_out // 2, affine=affine),
            PACT_with_log_quantize(base, time_step)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(C_out // 2, affine=affine),
            PACT_with_log_quantize(base, time_step)
        )

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        return out

