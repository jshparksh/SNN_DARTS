import torch
import torch.nn as nn
import math
from config import AugmentConfig
from quantization import PACT, PACT_with_log_quantize

args = AugmentConfig()

# v = sqrt
OPS_NOLOG = {
    'none' : lambda C, stride, affine: Zero_Nolog(stride),
    'max_pool_3x3' : lambda C, stride, affine: MaxPool_Nolog(3, stride=stride, padding=1),
    'skip_connect' : lambda C, stride, affine: Identity_Nolog() if stride == 1 else FactorizedReduce_Nolog(C, C, affine=affine),
    'conv_3x3' : lambda C, stride, affine: Conv_Nolog(C, C, 3, stride, 1, affine=affine),
    'sep_conv_3x3' : lambda C, stride, affine: SepConv_Nolog(C, C, 3, stride, 1, affine=affine),
    'dil_conv_3x3' : lambda C, stride, affine: DilConv_Nolog(C, C, 3, stride, 2, 2, affine=affine)
}

class ReLUConvBN_Nolog(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN_Nolog, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.op_type = 'rcb'
        
    def forward(self, x):
        self.flops = [args.batch_size / 4 * self.kernel_size * self.kernel_size * self.C_in * self.C_out * x.size()[2] * x.size()[3] / self.stride ** 2]
        return self.op(x)
    
class AvgPool_Nolog(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(AvgPool_Nolog, self).__init__()
        self.op = nn.Sequential(
            nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=False)
        )
        self.flops = [0]
        self.num_ifm = [1]
        self.non_zero_ifm = [torch.tensor(0)]
        self.op_type = 'avg'
    
    def forward(self, x):
        output = self.op(x)
        return output

    def spike_datas(self):
        return [0], [torch.tensor(0).cuda()], [torch.tensor(0).cuda()]

class MaxPool_Nolog(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(MaxPool_Nolog, self).__init__()
        self.op = nn.Sequential(
            nn.MaxPool2d(kernel_size, stride=stride, padding=padding),
            #PACT(alpha=args.init_log_alpha)
        )
        self.flops = [0]
        self.num_ifm = [1]
        self.non_zero_ifm = [torch.tensor(0)]
        self.op_type = 'max'
        
    def forward(self, x):
        output = self.op(x)
        #self.quan_infos = [[self.op[1].normed_ofm, self.op[1].base]]
        return output

    def spike_datas(self):
        #self.time_neuron = [torch.round(torch.where(quan_info[0] == 0, torch.tensor(0, dtype=torch.float32).cuda(), -torch.log(quan_info[0]))/torch.log(quan_info[1])) for quan_info in self.quan_infos]
        return [0], [torch.tensor(0).cuda()], [torch.tensor(0).cuda()] #self.time_neuron]
    
class Conv_Nolog(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Conv_Nolog, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
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
        return output
    
    def spike_datas(self):
        return [[0], [torch.tensor(0).cuda()], [torch.tensor(0).cuda()]]
    
class DilConv_Nolog(nn.Module):
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv_Nolog, self).__init__()    
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
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
        return output

    def spike_datas(self):
        return [[0], [torch.tensor(0).cuda()], [torch.tensor(0).cuda()]]
    
class SepConv_Nolog(nn.Module):
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv_Nolog, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
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
        return output

    def spike_datas(self):
        return [[0], [torch.tensor(0).cuda()], [torch.tensor(0).cuda()]]
    
class Identity_Nolog(nn.Module):

    def __init__(self):
        super(Identity_Nolog, self).__init__()
        self.op = nn.Sequential(
            PACT(alpha=args.init_log_alpha)
        )
        self.op_type = 'skip'
        
    def forward(self, x):
        output = x
        return output
    
    def spike_datas(self):
        return [[0], [torch.tensor(0).cuda()], [torch.tensor(0).cuda()]]
        
class Zero_Nolog(nn.Module):

    def __init__(self, stride):
        super(Zero_Nolog, self).__init__()
        self.stride = stride
        self.op_type = 'zero'
        
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        self.tensor_size = x[:,:,::self.stride,::self.stride].size()
        return x[:,:,::self.stride,::self.stride].mul(0.)
    
    def spike_datas(self):
        return [[0], [torch.tensor(0).cuda()], [torch.tensor(0).cuda()]]


class FactorizedReduce_Nolog(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce_Nolog, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

        # self.conv_1 = nn.Sequential(
        #     nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False),
        #     nn.BatchNorm2d(C_out // 2, affine=affine),
        #     PACT(alpha=args.init_log_alpha)
        # )
        # self.conv_2 = nn.Sequential(
        #     nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False),
        #     nn.BatchNorm2d(C_out // 2, affine=affine),
        #     PACT(alpha=args.init_log_alpha)
        # )
        self.C_in = C_in
        self.C_out = C_out
        self.flops = [0]
        self.num_ifm = [1]
        self.non_zero_ifm = [torch.tensor(0)]
        self.op_type = 'fr'
        
    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out

    def spike_datas(self):
        return [[0], [torch.tensor(0).cuda()], [torch.tensor(0).cuda()]]