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
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        y = torch.clamp(x, min=0., max=alpha.item())
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_variables
        
        # Gradient of x ( 0 <= x <= alpha )
        lt0      = x < 0
        gta      = x > alpha
        gi       = (~(lt0|gta)).float()
        grad_x   = grad_output*gi
        # Gradient of alpha ( x >= alpha )
        grad_alpha = torch.sum(grad_output*x.ge(alpha).float()).view(-1)
        return grad_x, grad_alpha
    
    
class PACT(nn.Module):
    def __init__(self, alpha=5.):
        super(PACT, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]), requires_grad=False)
        self.relu = nn.ReLU(inplace=False)
    def forward(self, input):
        qinput = pact_function.apply(input, self.alpha)
        return qinput

# edit here
class PACT_log_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, base, tmp_base, time_step, err_past):
        # Clamping with alpha
        c_x = torch.clamp(x, min=0., max=alpha.item()) # clamp activation value between 0 and alpha*base**(-1)
        normed_ofm = (c_x / alpha)
        log_value = torch.where(normed_ofm==0.0, torch.tensor(-time_step).float().cuda(), torch.log(normed_ofm)/torch.log(torch.tensor(base)))
        round = torch.round(log_value)
        q_y = torch.where(round > -time_step, base**round, torch.tensor(0.,).cuda()) * alpha
        err_curr = torch.sum(torch.abs((c_x-q_y))).view(-1)
        ctx.save_for_backward(x, q_y, alpha, base, round)
        ctx.constant = time_step
        return q_y, normed_ofm, err_curr

    
    @staticmethod
    def backward(ctx, grad_output, grad_normed_ofm, grad_err_curr):
        x, q_y, alpha, base, round = ctx.saved_variables 
        timestep = ctx.constant
        mina = alpha * (base**((-timestep+1-timestep)/2))
        lt0      = x < 0
        gta      = x > alpha
        gi       = (~(lt0|gta)).float()
        
        # grad_x = grad_output * gi
        # grad_alpha = torch.sum(grad_output*x.ge(mina)*x.lt(alpha)*base**round+grad_output*x.ge(alpha)).view(-1)
        # grad_tmp_base = torch.sum(grad_output*x.ge(mina)*x.lt(alpha)*alpha*round*base**(round-1)).view(-1)
        
        temper = 10 ##temperature of sigmoid
        sigmoid_f = 1/(1+torch.exp(-temper*(x/alpha-base**(-timestep+1))))
        grad_x = torch.where(x!=0, grad_output * gi, 0.0) ## LSQ
        grad_x += grad_output*(lt0 & x.gt(0)) * sigmoid_f*(1-sigmoid_f)*temper* base**(-timestep+1)
        grad_alpha = torch.sum(grad_output*x.ge(alpha).float()).view(-1) ##LSQ
        grad_alpha += torch.sum(grad_output*gi*(q_y-x)/alpha).view(-1) ##LSQ
        grad_alpha += torch.sum(grad_output*(lt0 & x.gt(0))*base**(-timestep+1)*(-sigmoid_f*(1-sigmoid_f)*x*temper+sigmoid_f)).view(-1) ##LSQ
        grad_tmp_base = torch.sum(grad_output*gi*q_y/base*round).view(-1) ##LSQ
        grad_tmp_base += torch.sum(grad_output*(lt0 & x.gt(0))*alpha*(timestep-1)*base**(-timestep)*(base**(-timestep+1)*temper*sigmoid_f*(1-sigmoid_f)-sigmoid_f)).view(-1) ##LSQ
        
        return grad_x, grad_alpha, None, grad_tmp_base, None, None
    
    
# edit here
# combined function which trains alpha and base together
class PACT_with_log_quantize(nn.Module):
    def __init__(self, alpha=5., base=2, time_step=4):
        super(PACT_with_log_quantize, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]), requires_grad=False)
        self.base = nn.Parameter(torch.Tensor([base]), requires_grad=False)
        self.tmp_base = nn.Parameter(torch.Tensor([0.]), requires_grad=False)
        self.time_step = time_step
        self.error = torch.tensor(0.0).cuda()
        
    def forward(self, input):
        qinput, normed_output, err_past = PACT_log_quantize.apply(input, self.alpha, self.base, self.tmp_base, self.time_step, self.error.clone().detach())
        self.normed_ofm = normed_output
        self.error = err_past
        
        return qinput

