import torch
import torch.nn as nn
import math
from config import SearchConfig
from quantization import PACT, PACT_with_log_quantize

args = SearchConfig()

# v = sqrt
OPS = {
    'none' : lambda C, stride, affine: Zero(stride),
    'max_pool_3x3' : lambda C, stride, affine: MaxPool(3, stride=stride, padding=1),
    'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'conv_3x3' : lambda C, stride, affine: Conv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine)
}

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            PACT_with_log_quantize()
        )
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.op_type = 'rcb'
        
    def forward(self, x):
        self.flops = [args.batch_size / 4 * self.kernel_size * self.kernel_size * self.C_in * self.C_out * x.size()[2] * x.size()[3] / self.stride ** 2]
        return self.op(x)

class AvgPool(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(AvgPool, self).__init__()
        self.op = nn.Sequential(
        nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=False),
        PACT_with_log_quantize()
        )
        self.flops = [0]
        self.num_ifm = [1]
        self.non_zero_ifm = [torch.tensor(0)]
        self.op_type = 'avg'
    
    def forward(self, x):
        output = self.op(x)
        self.quan_infos = [[self.op[1].normed_ofm, self.op[1].base]]
        self.num_ifm = [torch.numel(x)]
        self.non_zero_ifm = [torch.count_nonzero(x)]
        return output

    def spike_datas(self):
        self.spike_rate = [non_zero_ifm / num_ifm for non_zero_ifm, num_ifm in zip(self.non_zero_ifm, self.num_ifm)]
        # quan_info = [ofm, base]
        self.time_neuron = [torch.round(torch.where(quan_info[0] == 0, torch.tensor(0, dtype=torch.float32).cuda(), -torch.log(quan_info[0]))/torch.log(quan_info[1])) for quan_info in self.quan_infos]
        return [self.flops, self.spike_rate, self.time_neuron]

class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(MaxPool, self).__init__()
        self.op = nn.Sequential(
            nn.MaxPool2d(kernel_size, stride=stride, padding=padding),
            PACT_with_log_quantize()
        )
        self.flops = [0]
        self.num_ifm = [1]
        self.non_zero_ifm = [torch.tensor(0)]
        self.op_type = 'max'
        
    def forward(self, x):
        output = self.op(x)
        self.quan_infos = [[self.op[1].normed_ofm, self.op[1].base]]
        return output

    def spike_datas(self):
        self.time_neuron = [torch.round(torch.where(quan_info[0] == 0, torch.tensor(0, dtype=torch.float32).cuda(), -torch.log(quan_info[0]))/torch.log(quan_info[1])) for quan_info in self.quan_infos]
        return [[0], [torch.tensor(0).cuda()], self.time_neuron]
    
class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Conv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            PACT_with_log_quantize()
            )
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.flops = [0]
        self.num_ifm = [1]
        self.non_zero_ifm = [torch.tensor(0)]
        self.op_type = 'conv'
        
    def forward(self, x):
        output = self.op(x)
        self.quan_infos = [[self.op[2].normed_ofm, self.op[2].base]]
        self.flops = [args.batch_size / 4 * self.kernel_size * self.kernel_size * self.C_in * self.C_out * x.size()[2] * x.size()[3] / self.stride ** 2]
        self.num_ifm = [torch.numel(x)]
        self.non_zero_ifm = [torch.count_nonzero(x)]
        return output
    
    def spike_datas(self):
        self.spike_rate = [non_zero_ifm / num_ifm for non_zero_ifm, num_ifm in zip(self.non_zero_ifm, self.num_ifm)]
        self.time_neuron = [torch.round(torch.where(quan_info[0] == 0, torch.tensor(0, dtype=torch.float32).cuda(), -torch.log(quan_info[0]))/torch.log(quan_info[1])) for quan_info in self.quan_infos]
        return [self.flops, self.spike_rate, self.time_neuron]
    
class DilConv(nn.Module):
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()    
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            PACT_with_log_quantize(),
            nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            PACT_with_log_quantize()
            )
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.flops = [0]
        self.num_ifm = [1]
        self.non_zero_ifm = [torch.tensor(0)]
        self.op_type = 'dil'
        
    def forward(self, x):
        output = self.op(x)
        self.quan_infos = [[self.op[1].normed_ofm, self.op[1].base], [self.op[4].normed_ofm, self.op[4].base]]
        self.flops = [args.batch_size / 4 * self.kernel_size * self.kernel_size * self.C_in * self.C_in * x.size()[2] * x.size()[3] / self.stride ** 2 / self.C_in, # grouped conv -> FLOPS should be devided
                    args.batch_size / 4 * 1 * 1 * self.C_in * self.C_out * x.size()[2] * x.size()[3] / self.stride ** 2]
        self.num_ifm = [torch.numel(x), torch.numel(self.op[0:2](x))]
        self.non_zero_ifm = [torch.count_nonzero(x), torch.count_nonzero(self.op[0:2](x))]
        return output

    def spike_datas(self):
        self.spike_rate = [non_zero_ifm / num_ifm for non_zero_ifm, num_ifm in zip(self.non_zero_ifm, self.num_ifm)]
        self.time_neuron = [torch.round(torch.where(quan_info[0] == 0, torch.tensor(0, dtype=torch.float32).cuda(), -torch.log(quan_info[0]))/torch.log(quan_info[1])) for quan_info in self.quan_infos]
        return [self.flops, self.spike_rate, self.time_neuron]
    
class SepConv(nn.Module):
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            PACT_with_log_quantize(),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            PACT_with_log_quantize(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            PACT_with_log_quantize(),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            PACT_with_log_quantize()
        )
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.flops = [0]
        self.num_ifm = [1]
        self.non_zero_ifm = [torch.tensor(0)]
        self.op_type = 'sep'
        
    def forward(self, x):
        output = self.op(x)
        self.quan_infos = [[self.op[1].normed_ofm, self.op[1].base], [self.op[4].normed_ofm, self.op[4].base], [self.op[6].normed_ofm, self.op[6].base], [self.op[9].normed_ofm, self.op[9].base]]
        self.flops = [args.batch_size / 4 * self.kernel_size * self.kernel_size * self.C_in * self.C_in * x.size()[2] * x.size()[3] / self.stride ** 2 / self.C_in, # grouped conv -> FLOPS should be devided
                    args.batch_size / 4 * 1 * 1 * self.C_in * self.C_in * x.size()[2] * x.size()[3] / self.stride ** 2,
                    args.batch_size / 4 * self.kernel_size * self.kernel_size * self.C_in * self.C_in * x.size()[2] * x.size()[3] / self.stride ** 2 / self.C_in,
                    args.batch_size / 4 * 1 * 1 * self.C_in * self.C_out * x.size()[2] * x.size()[3] / self.stride ** 2]
        self.num_ifm = [torch.numel(x), torch.numel(self.op[0:2](x)), torch.numel(self.op[0:5](x)), torch.numel(self.op[0:7](x))]
        self.non_zero_ifm = [torch.count_nonzero(x), torch.count_nonzero(self.op[0:2](x)), torch.count_nonzero(self.op[0:5](x)), torch.count_nonzero(self.op[0:7](x))]
        return output

    def spike_datas(self):
        self.spike_rate = [non_zero_ifm / num_ifm for non_zero_ifm, num_ifm in zip(self.non_zero_ifm, self.num_ifm)]
        self.time_neuron = [torch.round(torch.where(quan_info[0] == 0, torch.tensor(0, dtype=torch.float32).cuda(), -torch.log(quan_info[0]))/torch.log(quan_info[1])) for quan_info in self.quan_infos]
        return [self.flops, self.spike_rate, self.time_neuron]
    
class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()
        self.op = nn.Sequential(
            PACT_with_log_quantize()
        )
        self.op_type = 'skip'
        
    def forward(self, x):
        output = self.op(x)
        self.quan_infos = [[self.op[0].normed_ofm, self.op[0].base]]
        return output
    
    def spike_datas(self):
        self.time_neuron = [torch.round(torch.where(quan_info[0] == 0, torch.tensor(0, dtype=torch.float32).cuda(), -torch.log(quan_info[0]))/torch.log(quan_info[1])) for quan_info in self.quan_infos]
        # assume input is output of spike neuron -> no need to calculate
        return [[0], [torch.tensor(0).cuda()], self.time_neuron]
        
class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
        self.op_type = 'zero'
        
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        self.tensor_size = x[:,:,::self.stride,::self.stride].size()
        return x[:,:,::self.stride,::self.stride].mul(0.)
    
    def spike_datas(self):
        return [[0], [torch.tensor(0).cuda()], [torch.tensor(0).cuda()]]


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(C_out // 2, affine=affine),
            PACT_with_log_quantize()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(C_out // 2, affine=affine),
            PACT_with_log_quantize()
        )
        self.C_in = C_in
        self.C_out = C_out
        self.flops = [0]
        self.num_ifm = [1]
        self.non_zero_ifm = [torch.tensor(0)]
        self.op_type = 'fr'
        
    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        self.flops = [args.batch_size / 4 * 1 * 1 * self.C_in * self.C_out // 2 * x.size()[2] * x.size()[3] / 2 ** 2,
                    args.batch_size / 4 * 1 * 1 * self.C_in * self.C_out // 2 * x.size()[2] * x.size()[3] / 2 ** 2]
        self.num_ifm = [torch.numel(x), torch.numel(x)]
        self.non_zero_ifm = [torch.count_nonzero(x), torch.count_nonzero(x)]
        self.ofms = [self.conv_1[2].normed_ofm, self.conv_2[2].normed_ofm]
        self.quan_infos = [[self.conv_1[2].normed_ofm, self.conv_1[2].base], [self.conv_2[2].normed_ofm, self.conv_2[2].base]]
        return out

    def spike_datas(self):
        self.spike_rate = [non_zero_ifm / num_ifm for non_zero_ifm, num_ifm in zip(self.non_zero_ifm, self.num_ifm)]
        self.time_neuron = [torch.round(torch.where(quan_info[0] == 0, torch.tensor(0, dtype=torch.float32).cuda(), -torch.log(quan_info[0]))/torch.log(quan_info[1])) for quan_info in self.quan_infos]
        return [self.flops, self.spike_rate, self.time_neuron]