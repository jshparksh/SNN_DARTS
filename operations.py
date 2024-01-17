import torch
import torch.nn as nn
import math
from config import SearchConfig
from quantization import PACT, PACT_with_log_quantize

args = SearchConfig()

# v = sqrt
OPS = {
    'none' : lambda C, stride, affine: Zero(stride),
    'max_pool_3x3_4v2' : lambda C, stride, affine: MaxPool(3, stride=stride, padding=1, base=math.sqrt(math.sqrt(2)), time_step=args.timestep),
    'max_pool_3x3_v2' : lambda C, stride, affine: MaxPool(3, stride=stride, padding=1, base=math.sqrt(2), time_step=args.timestep),
    'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, base=math.sqrt(2), time_step=args.timestep, affine=affine),
    'conv_3x3_4v2' : lambda C, stride, affine: Conv(C, C, 3, stride, 1, base=math.sqrt(math.sqrt(2)), time_step=args.timestep, affine=affine),
    'conv_3x3_v2' : lambda C, stride, affine: Conv(C, C, 3, stride, 1, base=math.sqrt(2), time_step=args.timestep, affine=affine),
    'sep_conv_3x3_4v2' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, base=math.sqrt(math.sqrt(2)), time_step=args.timestep, affine=affine),
    'sep_conv_3x3_v2' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, base=math.sqrt(2), time_step=args.timestep, affine=affine),
    'dil_conv_3x3_4v2' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, base=math.sqrt(math.sqrt(2)), time_step=args.timestep, affine=affine),
    'dil_conv_3x3_v2' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, base=math.sqrt(2), time_step=args.timestep, affine=affine)
}

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            PACT()
        )
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        self.flops = [self.kernel_size * self.kernel_size * self.C_in * self.C_out * x.size()[2] * x.size()[3] / self.stride ** 2]
        return self.op(x)

class AvgPool(nn.Module):
    def __init__(self, kernel_size, stride, padding, base, time_step):
        super(AvgPool, self).__init__()
        self.op = nn.Sequential(
        nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=False),
        PACT_with_log_quantize(base, time_step)
        )
        self.base = base
        self.flops = [0]
        self.num_ifm = [1]
        self.non_zero_ifm = [0]
        self.ofms = [torch.tensor(1)]
    
    def forward(self, x):
        self.num_ifm = [torch.numel(x)]
        self.non_zero_ifm = [torch.count_nonzero(x)]
        self.ofms = [self.op(x)]
        return self.ofms

    def spike_datas(self):
        self.scale = [self.op[1].alpha]
        self.spike_rate = [non_zero_ifm / num_ifm for non_zero_ifm, num_ifm in zip(self.non_zero_ifm, self.num_ifm)]
        self.time_neuron = [-torch.log(ofm/scale)/torch.log(torch.tensor(self.base)) for ofm, scale in zip(self.ofms, self.scale)]
        return [self.flops, self.spike_rate, self.time_neuron]

class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding, base, time_step):
        super(MaxPool, self).__init__()
        self.op = nn.Sequential(
        nn.MaxPool2d(kernel_size, stride=stride, padding=padding),
        PACT_with_log_quantize(base, time_step)
        )
        self.base = base
        self.flops = [0]
        self.num_ifm = [1]
        self.non_zero_ifm = [0]
        self.ofms = [torch.tensor(1)]
        
    def forward(self, x):
        self.num_ifm = [torch.numel(x)]
        self.non_zero_ifm = [torch.count_nonzero(x)]
        self.ofms = [self.op(x)]
        return self.op(x)

    def spike_datas(self):
        self.scale = [self.op[1].alpha]
        self.spike_rate = [non_zero_ifm / num_ifm for non_zero_ifm, num_ifm in zip(self.non_zero_ifm, self.num_ifm)]
        self.time_neuron = [-torch.log(ofm/scale)/torch.log(torch.tensor(self.base)) for ofm, scale in zip(self.ofms, self.scale)]
        return [self.flops, self.spike_rate, self.time_neuron]
    
class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, base, time_step, affine=True):
        super(Conv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            PACT_with_log_quantize(base, time_step)
            )
        self.base = base
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.flops = [0]
        self.num_ifm = [1]
        self.non_zero_ifm = [0]
        self.ofms = [torch.tensor(1)]
        
    def forward(self, x):
        self.flops = [self.kernel_size * self.kernel_size * self.C_in * self.C_out * x.size()[2] * x.size()[3] / self.stride ** 2]
        self.num_ifm = [torch.numel(x)]
        self.non_zero_ifm = [torch.count_nonzero(x)]
        self.ofms = [self.op(x)]
        return self.op(x)
    
    def spike_datas(self):
        self.scale = [self.op[2].alpha]
        self.spike_rate = [non_zero_ifm / num_ifm for non_zero_ifm, num_ifm in zip(self.non_zero_ifm, self.num_ifm)]
        self.time_neuron = [-torch.log(ofm/scale)/torch.log(torch.tensor(self.base)) for ofm, scale in zip(self.ofms, self.scale)]
        return [self.flops, self.spike_rate, self.time_neuron]
    
class DilConv(nn.Module):
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, base, time_step, affine=True):
        super(DilConv, self).__init__()    
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            PACT_with_log_quantize(base, time_step),
            nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            PACT_with_log_quantize(base, time_step)
            )
        self.base = base
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.flops = [0]
        self.num_ifm = [1]
        self.non_zero_ifm = [0]
        self.ofms = [torch.tensor(1)]
        
    def forward(self, x):
        self.flops = [(self.kernel_size * self.kernel_size * self.C_in * self.C_in * x.size()[2] * x.size()[3] / self.stride ** 2 / self.C_in), # grouped conv -> FLOPS should be devided
                    (1 * 1 * self.C_in * self.C_out * x.size()[2] * x.size()[3] / self.stride ** 2)]
        self.num_ifm = [torch.numel(x), torch.numel(self.op[0](x))]
        self.non_zero_ifm = [torch.count_nonzero(x), torch.count_nonzero(self.op[0](x))]
        self.ofms = [self.op[0:2](x), self.op(x)]
        return self.op(x)

    def spike_datas(self):
        self.scale = [self.op[1].alpha, self.op[4].alpha]
        self.spike_rate = [non_zero_ifm / num_ifm for non_zero_ifm, num_ifm in zip(self.non_zero_ifm, self.num_ifm)]
        self.time_neuron = [-torch.log(ofm/scale)/torch.log(torch.tensor(self.base)) for ofm, scale in zip(self.ofms, self.scale)]
        return [self.flops, self.spike_rate, self.time_neuron]
    
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
        self.base = base
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.flops = [0]
        self.num_ifm = [1]
        self.non_zero_ifm = [0]
        self.ofms = [torch.tensor(1)]
        
    def forward(self, x):
        self.flops = [(self.kernel_size * self.kernel_size * self.C_in * self.C_in * x.size()[2] * x.size()[3] / self.stride ** 2 / self.C_in), # grouped conv -> FLOPS should be devided
                    (1 * 1 * self.C_in * self.C_in * x.size()[2] * x.size()[3] / self.stride ** 2 / self.C_in),
                    (self.kernel_size * self.kernel_size * self.C_in * self.C_in * x.size()[2] * x.size()[3] / self.stride ** 2 / self.C_in),
                    (1 * 1 * self.C_in * self.C_out * x.size()[2] * x.size()[3] / self.stride ** 2)]
        self.num_ifm = [torch.numel(x), torch.numel(self.op[0](x)), torch.numel(self.op[0:3](x)), torch.numel(self.op[0:6](x))]
        self.non_zero_ifm = [torch.count_nonzero(x), torch.count_nonzero(self.op[0](x)), torch.count_nonzero(self.op[0:3](x)), torch.count_nonzero(self.op[0:6](x))]
        self.ofms = [self.op[0:2](x), self.op[0:5](x), self.op[0:7](x), self.op(x)]
        return self.op(x)

    def spike_datas(self):
        self.scale = [self.op[1].alpha, self.op[4].alpha, self.op[6].alpha, self.op[9].alpha]
        self.spike_rate = [non_zero_ifm / num_ifm for non_zero_ifm, num_ifm in zip(self.non_zero_ifm, self.num_ifm)]
        self.time_neuron = [-torch.log(ofm/scale)/torch.log(torch.tensor(self.base)) for ofm, scale in zip(self.ofms, self.scale)]
        return [self.flops, self.spike_rate, self.time_neuron]
    
class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        self.device = x.device
        return x
    
    def spike_datas(self):
        # assume input is output of spike neuron -> no need to calculate
        return [[0], [0], [torch.tensor(0, dtype=torch.float32, requires_grad=True).to(self.device)]]


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
        
    def forward(self, x):
        self.device = x.device
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)
    
    def spike_datas(self):
        return [[0], [0], [torch.tensor(0, dtype=torch.float32, requires_grad=True).to(self.device)]]


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
        self.base = base
        self.C_in = C_in
        self.C_out = C_out
        self.flops = [0]
        self.num_ifm = [1]
        self.non_zero_ifm = [0]
        self.ofms = [torch.tensor(1)]
        
    def forward(self, x):
        x = self.relu(x)
        self.flops = [(1 * 1 * self.C_in * self.C_out // 2 * x.size()[2] * x.size()[3] / 2 ** 2),
                    (1 * 1 * self.C_in * self.C_out // 2 * x.size()[2] * x.size()[3] / 2 ** 2)]
        self.num_ifm = [torch.numel(self.conv_1[0](x)), torch.numel(self.conv_2[0](x))]
        self.non_zero_ifm = [torch.count_nonzero(self.conv_1[0](x)), torch.count_nonzero(self.conv_2[0](x))]
        self.ofms = [self.conv_1(x), self.conv_2(x)]
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        return out

    def spike_datas(self):
        self.scale = [self.conv_1[2].alpha, self.conv_2[2].alpha]
        self.spike_rate = [non_zero_ifm / num_ifm for non_zero_ifm, num_ifm in zip(self.non_zero_ifm, self.num_ifm)]
        self.time_neuron = [-torch.log(ofm/scale)/torch.log(torch.tensor(self.base)) for ofm, scale in zip(self.ofms, self.scale)]
        return [self.flops, self.spike_rate, self.time_neuron]