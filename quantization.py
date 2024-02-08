import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd.function import InplaceFunction
from torch.autograd import Variable

"""
    Make sure that input is clipped between [0. inf]

    ==> In pact_function this clipping function exists.
    Otherwise, Quantized value will be strange
"""

class pact_function(InplaceFunction):
    # See https://github.com/obilaniu/GradOverride/blob/master/functional.py
    # See https://arxiv.org/pdf/1805.06085.pdf
    @staticmethod
    def forward(ctx, x, alpha, base, time_step):
        ctx.save_for_backward(x, alpha)
        # tensor.clamp(min=0., max=alpha) <- alpha is parameter (torch.tensor) 
        # clamp min, max should not be tensor, so use tensor.min(alpha)
        """same to : PACT function y = 0.5 * (torch.abs(x) - torch.abs(x - alpha) + alpha)"""
        y = torch.clamp(x, min=alpha.item()*base**(-time_step-1), max=alpha.item()) #alpha.item()*base**(-time_step - 1), max = alpha.item())
        return y
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_variables
        """
        Same to : grad_input[x < 0] = 0.
                  grad_input[alpha < x] = 0.
        """
        # Gradient of x ( 0 <= x <= alpha )
        lt0      = x < 0
        gta      = x > alpha
        gi       = (~(lt0|gta)).float()
        grad_x   = grad_output*gi
        # Gradient of alpha ( x >= alpha )
        grad_alpha = torch.sum(grad_output*x.ge(alpha).float()).view(-1)
        #grad_alpha = grad_output.new([ga])
        return grad_x, grad_alpha, None, None

class log_quantize(InplaceFunction):
    @staticmethod
    def forward(ctx, x, base, time_step, scale): # x = normed_ofm
        x = x / scale
        log_value = torch.log(x)/torch.log(torch.tensor(base))
        round = torch.where(log_value <= 0.0, torch.round(log_value), torch.tensor(0.).cuda())
        return torch.where(round > -time_step, base**round, torch.tensor(0.,).cuda()) * scale
#        assert False, x.view(-1)[torch.argmax(x-torch.where(round > -time_step, base**round, torch.tensor(0.,).cuda()))]
       
#        assert False, torch.max(x-torch.where(round > -time_step, base**round, torch.tensor(0.,).cuda()))
    
    @staticmethod
    def backward(ctx, grad_output):
        # STE 
        grad_x = grad_output.clone()
        return grad_x, None, None, None

class PACT(nn.Module):
    def __init__(self, alpha=5.):
        super(PACT, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]), requires_grad=True)
        self.relu = nn.ReLU(inplace=False)
    def forward(self, input):
        qinput = pact_function.apply(input, self.alpha, 0, -3)
        return qinput
    
class PACT_with_log_quantize(nn.Module):
    def __init__(self, base, time_step, alpha=5.):
        super(PACT_with_log_quantize, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]), requires_grad=True)
        self.relu = nn.ReLU(inplace=False)
        self.base = base
        self.time_step = time_step
        self.max_value = 0
    
    def forward(self, input):
        qinput = pact_function.apply(input, self.alpha, self.base, self.time_step)
        self.normed_ofm = (qinput / self.alpha)
        qinput = log_quantize.apply(qinput, self.base, self.time_step, self.alpha)
        return qinput
