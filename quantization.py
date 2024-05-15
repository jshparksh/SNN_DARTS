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
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        y = torch.clamp(x, min=0., max=alpha.item()) #alpha.item()*base.item()**(-time_step), max=alpha.item())
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
        return grad_x, grad_alpha
    
# class log_quantize(InplaceFunction):
#     @staticmethod
#     def forward(ctx, x, alpha, base, tmp_base, time_step, err_past):
#         normed_ofm = (x / alpha)
#         log_value = torch.where(normed_ofm==0.0, torch.tensor(-time_step).float().cuda(), torch.log(normed_ofm)/torch.log(torch.tensor(base)))
#         round = torch.round(log_value)
#         q_y = torch.where(round > -time_step, base**round, torch.tensor(0.,).cuda()) * alpha
#         err_curr = torch.sum(torch.abs((x-q_y))).view(-1)
#         ctx.save_for_backward(x, alpha, base, round, err_past, err_curr)
#         ctx.constant = time_step
#         return q_y, err_curr
    
#     @staticmethod
#     def backward(ctx, grad_output, grad_error):
#         x, alpha, base, round, err_past, err_curr = ctx.saved_variables
#         min_act = alpha * (base**(-ctx.constant)+base**(1-ctx.constant))/2
#         gtm      = x > min_act
#         lta      = x < alpha
#         gta      = x > alpha
#         grad_x = grad_output * gtm
#         grad_alpha = torch.sum(grad_output * x.ge(min_act) * x.lt(alpha) * base**round).view(-1)
#         grad_tmp_base = torch.sum(grad_output * x.ge(min_act) * x.lt(alpha) * alpha * round * base**(round-1)).view(-1)
#         # if err_past != 0:
#         #     grad_tmp_base = torch.sum(grad_output*(err_curr-err_past)).view(-1)
#         # else:
#         #     grad_tmp_base = torch.sum(grad_output*0).view(-1)
#         return grad_x, grad_alpha, None, grad_tmp_base, None, None
    
    
class PACT(nn.Module):
    def __init__(self, alpha=5.):
        super(PACT, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]), requires_grad=False)
        self.relu = nn.ReLU(inplace=False)
    def forward(self, input):
        qinput = pact_function.apply(input, self.alpha, 0, -3)
        return qinput

# # edit here
# # combined function which trains alpha and base together
# class PACT_with_log_quantize(nn.Module):
#     def __init__(self, alpha=3., base=2.0, time_step=4):
#         super(PACT_with_log_quantize, self).__init__()
#         self.alpha = nn.Parameter(torch.Tensor([alpha]), requires_grad=False)
#         self.base = nn.Parameter(torch.Tensor([base]), requires_grad=False)
#         self.tmp_base = nn.Parameter(torch.Tensor([0.]), requires_grad=False)
#         self.time_step = time_step
#         self.error = torch.tensor(0.0).cuda()
        
#     def forward(self, input):
#         qinput = pact_function.apply(input, self.alpha)
#         self.normed_ofm = qinput / self.alpha
#         qinput, err_past = log_quantize.apply(qinput, self.alpha, self.base, self.tmp_base, self.time_step, self.error.clone().detach())
#         self.error = err_past
#         return qinput


# edit here
class PACT_log_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, base, tmp_base, time_step, err_past):
        c_x = torch.clamp(x, min=0., max=alpha.item()*base.item()**(-1)) # clamp activation value between 0 and alpha*base**(-1)
        normed_ofm = (c_x / (alpha*base**(-1)))
        log_value = torch.where(normed_ofm==0.0, torch.tensor(-time_step).float().cuda(), torch.log(normed_ofm)/torch.log(torch.tensor(base)))
        round = torch.round(log_value)
        q_y = torch.where(round > -time_step, base**round, torch.tensor(0.,).cuda()) * alpha*base**(-1)
        err_curr = torch.sum(torch.abs((c_x-q_y))).view(-1)
        ctx.save_for_backward(x, log_value, alpha, base, round, err_past, err_curr)
        ctx.constant = time_step
        return q_y, normed_ofm, err_curr

        # y = torch.clamp(x, min=0., max=alpha.item())
        # normed_ofm = (y / alpha)
        # log_value = torch.where(normed_ofm==0.0, torch.tensor(-time_step).float().cuda(), torch.log(normed_ofm)/torch.log(torch.tensor(base)))
        # round = torch.round(log_value)
        # q_y = torch.where(round > -time_step, base**round, torch.tensor(0.,).cuda()) * alpha
        # err_curr = torch.sum(torch.abs((x-q_y))).view(-1)
        # ctx.save_for_backward(x, alpha, base, err_past, err_curr)
        # ctx.constant = time_step
        # return q_y, normed_ofm, err_curr
    
    @staticmethod
    def backward(ctx, grad_output, grad_normed_ofm, grad_err_curr):
        x, log_value, alpha, base, round, err_past, err_curr = ctx.saved_variables #round value can be lower than time_step
        
        min_act = alpha * base**(-1) * (base**(-ctx.constant)+base**(1-ctx.constant))/2
        max_act = alpha * base**(-1) * (base**0+base**(-1))/2
        ltmin      = x < min_act
        gemax      = x >= max_act
        gi       = (~(ltmin|gemax)).float()
        # gi         = (~ltmin).float()
        
        # STE
        grad_x = torch.where(x==0, torch.tensor(0.,).cuda(), grad_output * gi * alpha * base**(round-1) / x)
        grad_alpha = torch.sum(grad_output * x.ge(max_act) * base**(-1)).view(-1)
        grad_tmp_base = torch.sum(grad_output * x.ge(min_act) * x.lt(max_act) * alpha * base**(round-2) * (-log_value + round) - grad_output * x.ge(max_act) * alpha / (base**2)).view(-1)
        
        # 0514
        # grad_x = grad_output * gi
        # grad_alpha = torch.sum(grad_output * x.ge(min_act) * x.lt(max_act) * base**(round-1) + grad_output * x.ge(max_act) * base**(-1)).view(-1)
        # grad_tmp_base = torch.sum(grad_output * x.ge(min_act) * x.lt(max_act) * alpha * (round-1) * base**(round-2) - grad_output * x.ge(max_act) * alpha / (base**2)).view(-1)
        
        
        # grad_alpha = torch.sum(grad_output * x.ge(min_act) * x.lt(alpha) * base**round + grad_output * x.ge(alpha)).view(-1)
        # grad_tmp_base = torch.sum(grad_output * x.ge(min_act) * x.lt(alpha) * alpha * round * base**(round-1)).view(-1)
        # if err_past != 0:
        #     grad_tmp_base = torch.sum(grad_output*(err_curr-err_past)).view(-1)
        # else:
        #     grad_tmp_base = torch.sum(grad_output*0).view(-1)
        return grad_x, grad_alpha, None, grad_tmp_base, None, None
    
# edit here
# combined function which trains alpha and base together
class PACT_with_log_quantize(nn.Module):
    def __init__(self, alpha=6., base=2.0, time_step=8):
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

