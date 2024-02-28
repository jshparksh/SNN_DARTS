import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
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
        y = torch.clamp(x, min=alpha.item()*base.item()**(-time_step), max=alpha.item()) #min=alpha.item()*base**(-time_step-1), max=alpha.item())
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
    
class test_pact_function(InplaceFunction):
    # See https://github.com/obilaniu/GradOverride/blob/master/functional.py
    # See https://arxiv.org/pdf/1805.06085.pdf
    @staticmethod
    def forward(ctx, x, alpha, base, time_step):
        ctx.save_for_backward(x, alpha, base)
        ctx.constant = time_step
        # tensor.clamp(min=0., max=alpha) <- alpha is parameter (torch.tensor) 
        # clamp min, max should not be tensor, so use tensor.min(alpha)
        """same to : PACT function y = 0.5 * (torch.abs(x) - torch.abs(x - alpha) + alpha)"""
        x = torch.clamp(x, min=0., max=alpha.item()) #min=alpha.item()*base.item()**(-time_step), max=alpha.item()) # 
        return x
        x = x / alpha
        log_value = torch.where(x==0.0, torch.tensor(-time_step).float().cuda(), torch.log(x)/torch.log(torch.tensor(base)))
        round = torch.where(log_value <= 0.0, torch.round(log_value), torch.tensor(0.).cuda())
        return torch.where(round > -time_step, base**round, torch.tensor(0.,).cuda()) * alpha
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, base = ctx.saved_variables
        """
        Same to : grad_input[x < 0] = 0.
                  grad_input[alpha < x] = 0.
        """
        # Gradient of x ( 0 <= x <= alpha )
        min_act = alpha*(base**(-ctx.constant)+base**(-1-ctx.constant))/2
        lt0      = x < 0
        gta      = x > alpha
        gi       = (~(lt0|gta)).float()
        grad_x   = grad_output*gi
        # Gradient of alpha ( x >= alpha )
        #grad_alpha = torch.tensor(0.,).cuda().view(-1)
        grad_alpha = torch.sum(grad_output*x.ge(alpha).float()).view(-1)
        #grad_base = torch.sum(grad_output*(x.lt(min_act).float())).view(-1)
        grad_base = torch.sum(torch.where(x<min_act, grad_x, grad_x*x*(1-base)/(math.sqrt(base)*(1+base)**2))).view(-1)
        #grad_base = torch.sum(grad_output*(((1-base)/(1+base))*x.ge(min_act).float()+1)).view(-1)
        #grad_alpha = grad_output.new([ga])
        return grad_x, grad_alpha, grad_base, None
    
class test_log_quantize(InplaceFunction):
    @staticmethod
    def forward(ctx, x, scale, base, time_step): # x = normed_ofm
        ctx.save_for_backward(x, scale, base)
        ctx.constant = time_step
        x = x / scale
        log_value = torch.where(x==0.0, torch.tensor(-time_step).float().cuda(), torch.log(x)/torch.log(torch.tensor(base)))
        round = torch.where(log_value <= 0.0, torch.round(log_value), torch.tensor(0.).cuda())
        return torch.where(round > -time_step, base**round, torch.tensor(0.,).cuda()) * scale
#        assert False, x.view(-1)[torch.argmax(x-torch.where(round > -time_step, base**round, torch.tensor(0.,).cuda()))]
       
#        assert False, torch.max(x-torch.where(round > -time_step, base**round, torch.tensor(0.,).cuda()))
    
    @staticmethod
    def backward(ctx, grad_output):
        x, scale, base = ctx.saved_variables
        min_act = scale*(base**(-ctx.constant)+base**(-1-ctx.constant))/2
        lt0      = x < 0
        gta      = x > scale
        gi       = (~(lt0|gta)).float()
        grad_x   = grad_output*gi
        grad_base = torch.sum(torch.where(x<min_act, grad_x, grad_x*x*(1-base)/(math.sqrt(base)*(1+base)**2))).view(-1)
        return grad_x, None, grad_base, None
    
class log_quantize(InplaceFunction):
    @staticmethod
    def forward(ctx, x, scale, base, time_step): # x = normed_ofm
        x = x / scale
        log_value = torch.where(x==0.0, torch.tensor(-time_step).float().cuda(), torch.log(x)/torch.log(torch.tensor(base)))
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

class Test_PACT(nn.Module):
    def __init__(self, alpha=5., base=2., time_step=4):
        super(Test_PACT, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]), requires_grad=True)
        self.base = nn.Parameter(torch.Tensor([base]), requires_grad=True)
        self.time_step = time_step
        
    def forward(self, input):
        qinput = pact_function.apply(input, self.alpha, self.base, self.time_step)
        qinput = test_log_quantize.apply(qinput, self.alpha, self.base, self.time_step)
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
        qinput = log_quantize.apply(qinput, self.alpha, self.base, self.time_step)
        return qinput
