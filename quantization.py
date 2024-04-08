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
    def forward(ctx, x, alpha, time_step):
        ctx.save_for_backward(x, alpha)
        ctx.constant = time_step
        
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
        return grad_x, grad_alpha, None
    
class log_quantize(InplaceFunction):
    @staticmethod
    def forward(ctx, x, scale, base, tmp_base, time_step):
        x = x / scale
        log_value = torch.where(x==0.0, torch.tensor(-time_step).float().cuda(), torch.log(x)/torch.log(torch.tensor(base)))
        round = torch.round(log_value)

        ctx.save_for_backward(x, scale, base, round)
        ctx.constant = time_step
        return torch.where(round > -time_step, base**round, torch.tensor(0.,).cuda()) * scale
    
    @staticmethod
    def backward(ctx, grad_output):
        x, scale, base, round = ctx.saved_variables

        q_x = torch.where(round > -ctx.constant, base**round, torch.tensor(0.,).cuda())
        
        min_act = (base**(-ctx.constant)+base**(1-ctx.constant))/2
        grad_x = grad_output*(x>min_act)

        m =torch.tensor(1e-6).cuda()
        
        # assert False, torch.sum((torch.where(x < min_act, -2*q_x*(x-q_x)/base*(torch.log(m)/torch.log(base)), -2*q_x*(x-q_x)/base*(round-(torch.log(x)/torch.log(base)))))
        # n_non_zero = torch.count_nonzero(x).float()
        grad_base = torch.sum(torch.where(x < min_act, -2*q_x*(x-q_x)/base*(torch.log(m)/torch.log(base)), -2*q_x*(x-q_x)/base*(round-(torch.log(x)/torch.log(base))))).view(-1)
        # if n_non_zero == 0:
        #     grad_base = torch.mean(torch.where(x < min_act, -2*q_x*(x-q_x)/base*(torch.log(m)/torch.log(base)), -2*q_x*(x-q_x)/base*(round-(torch.log(x)/torch.log(base))))).view(-1)
        # else:
        #     grad_base = (torch.sum(torch.where(x < min_act, -2*q_x*(x-q_x)/base*(torch.log(m)/torch.log(base)), -2*q_x*(x-q_x)/base*(round-(torch.log(x)/torch.log(base))))).view(-1))/n_non_zero
        # grad_base_2 = (torch.where(x < min_act, -0*q_x*(x-q_x)*, -2*q_x*(x-q_x)*(round-torch.log(x))/torch.log(base))).view(-1)
        return grad_x, None, None, grad_base, None
        
        # k = 1000
        # time = ctx.constant
        # x = x / scale
        # min_act = (base**(-time)+base**(1-time))/2
        # tmp_grad_x = 0
        # tmp_grad_base = 0
        # for t in range(1, time+1):
        #     x_exp = torch.exp(-k*(x-((base**(-t+1))+(base**(-t)))/2))
        #     if time == t:
        #         tmp_grad_x += (k*(((base**(1-t))))*x_exp)/(((x_exp+1)**2))
        #         tmp_grad_base += k*(((base**(1-t)))*(t*(base**(-1-t))-(1-t)*(base**(-t)))*x_exp)/(2*((x_exp+1)**2)) + ((1-t)*((base**(-t))))/(x_exp+1)
        #     else:
        #         tmp_grad_x += (k*(((base**(1-t))-(base**(-t))))*x_exp)/(((x_exp+1)**2))
        #         tmp_grad_base += k*(((base**(1-t))-(base**(-t)))*(t*(base**(-1-t))-(1-t)*(base**(-t)))*x_exp)/(2*((x_exp+1)**2)) + (t*(base**(-1-t))+(1-t)*(base**(-t)))/(x_exp+1)

        # grad_x   = grad_output#*tmp_grad_x
        
        # grad_base = torch.sum(grad_output*tmp_grad_base).view(-1) #tmp_grad_base
        # #grad_base = torch.sum(torch.where(x<min_act,))
        # #grad_base = torch.sum(torch.where(x==0.0, grad_output*0, gtr.float()*grad_output*scale*round*base**(round-1)*torch.log(x/scale)/torch.log(base))).view(-1)
        
        # return grad_x, None, grad_base, None

# edit here
class PACT_log_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, base, tmp_base, time_step):
        y = torch.clamp(x, min=0., max=alpha.item())
        normed_ofm = (y / alpha)
        log_value = torch.where(normed_ofm==0.0, torch.tensor(-time_step).float().cuda(), torch.log(normed_ofm)/torch.log(torch.tensor(base)))
        round = torch.round(log_value)
        ctx.save_for_backward(x, alpha, base, round)
        ctx.constant = time_step
        y = torch.where(round > -time_step, base**round, torch.tensor(0.,).cuda()) * alpha
        return y, normed_ofm
    
    @staticmethod
    def backward(ctx, grad_output, grad_normed_ofm):
        x, alpha, base, round = ctx.saved_variables
        
        min_act = (base**(-ctx.constant)+base**(1-ctx.constant))/2
        lt0      = x < 0
        ltm      = x < min_act
        gta      = x > alpha
        gi       = (~(ltm|gta)).float()
        
        grad_x   = grad_output * gi
        grad_alpha = torch.sum(grad_output*x.ge(alpha).float()).view(-1)
        
        
        x = torch.clamp(x, min=0., max=alpha.item())
        x = x / alpha
        q_x = torch.where(round > -ctx.constant, base**round, torch.tensor(0.,).cuda())
        m =torch.tensor(1e-6).cuda()
        grad_tmp_base = torch.sum(torch.where(x < min_act, -2*q_x*(x-q_x)/base*(torch.log(m))/torch.log(base), (base>1)*-2*q_x*(x-q_x)/base*(round-torch.log(x))/(torch.log(base)))).view(-1)
        #grad_base = torch.sum(x*(x-q_x)*base**(-2/3)*(x>0)).view(-1)
        #grad_base = torch.sum(-(x - q_x)*x*(x>0)).view(-1)/torch.sqrt(base)/alpha
        return grad_x, grad_alpha, None, grad_tmp_base, None 
    
        # min_act = alpha*(base**(-ctx.constant)+base**(1-ctx.constant))/2
        # m = 1e-1
        # lt0      = x < 0
        # ltm      = x < min_act
        # gt0      = x > 0
        # gtm      = x > min_act
        # gta      = x > alpha
        # gi       = (~(lt0|gta)).float()
        # grad_x   = grad_output * gi #* math.sqrt(base)
        # grad_alpha = torch.sum(grad_output*x.ge(alpha).float()).view(-1) 
        # # edit here
        # grad_base = torch.sum(torch.where(x<min_act, -torch.abs(grad_output)*2*x*base**(-ctx.constant)*(base*ctx.constant+ctx.constant-1)/((base+1)**2*m)*(x>0),
        #                                   torch.abs(grad_output)*2*x/(base+1)**2)).view(-1) #*(1-base)*x/(math.sqrt(base)*(1+base)**2))).view(-1)
        #grad_base = torch.sum(torch.where(x<min_act, -grad_output*(x>0).float(), 1/2*grad_output*x/base**(1/2))).view(-1) # grad_output*x*(1-base)/(math.sqrt(base)*(1+base)**2)
        #grad_base = torch.sum(1/2*x.ge(min_act).float()*x.le(alpha).float()*grad_output*x/base**(1/2)).view(-1)
        #grad_base = torch.sum(torch.where(x<min_act, -grad_output*(x>0).float(), 1/2*grad_output/math.sqrt(base))).view(-1) # grad_output*x*(1-base)/(math.sqrt(base)*(1+base)**2)
        #grad_base = torch.sum(torch.where(x<min_act, grad_output*0, grad_output*0)).view(-1)


class PACT(nn.Module):
    def __init__(self, alpha=5.):
        super(PACT, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]), requires_grad=False)
        self.relu = nn.ReLU(inplace=False)
    def forward(self, input):
        qinput = pact_function.apply(input, self.alpha, 0, -3)
        return qinput

"""
# edit here
# split training alpha and base with seperate functions
class PACT_with_log_quantize(nn.Module):
    def __init__(self, alpha=5., base=2.0, time_step=4):
        super(PACT_with_log_quantize, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]), requires_grad=False)
        self.base = nn.Parameter(torch.Tensor([base]), requires_grad=False)
        self.tmp_base = nn.Parameter(torch.Tensor([0.]), requires_grad=False)
        self.time_step = time_step
        
    def forward(self, input):
        qinput = pact_function.apply(input, self.alpha, self.time_step)
        self.normed_ofm = (qinput / self.alpha)
        #qinput = log_quantize.apply(qinput, self.alpha, self.base, self.tmp_base, self.time_step)
        return qinput
    
"""
# edit here
# combined function which trains alpha and base together
class PACT_with_log_quantize(nn.Module):
    def __init__(self, alpha=5., base=2.0, time_step=4):
        super(PACT_with_log_quantize, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]), requires_grad=False)
        self.base = nn.Parameter(torch.Tensor([base]), requires_grad=False)
        self.tmp_base = nn.Parameter(torch.Tensor([0.]), requires_grad=False)
        self.time_step = time_step
        
    def forward(self, input):
        qinput, normed_output = PACT_log_quantize.apply(input, self.alpha, self.base, self.tmp_base, self.time_step)
        self.normed_ofm = normed_output
        return qinput



""" 
class PACT_log_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, base, time_step):
        ctx.save_for_backward(x, alpha, base)
        ctx.constant = time_step
        y = torch.clamp(x, min=0., max=alpha.item())
        normed_ofm = (y / alpha)
        log_value = torch.where(normed_ofm==0.0, torch.tensor(-time_step).float().cuda(), torch.log(normed_ofm)/torch.log(torch.tensor(base)))
        round = torch.round(log_value)
        #round = torch.where((log_value <= 0.0), torch.round(log_value), torch.tensor(0.).cuda())
        y = torch.where(round > -time_step, base**round, torch.tensor(0.,).cuda()) * alpha
        return y, normed_ofm
    
    @staticmethod
    def backward(ctx, grad_output, grad_normed_ofm):
        x, alpha, base = ctx.saved_variables
        min_act = alpha*(base**(-ctx.constant)+base**(1-ctx.constant))/2
        
        lt0      = x < 0
        ltm      = x < min_act
        gt0      = x > 0
        gtm      = x > min_act
        gta      = x > alpha
        gi       = (~(lt0|gta)).float()
        grad_x   = grad_output*gi
        grad_alpha = torch.sum(grad_output*x.ge(alpha).float()).view(-1)
        grad_base = torch.sum(torch.where(x<min_act, grad_output*(x>0).float(), 1/2*(base>1).float()*grad_output/math.sqrt(base))).view(-1) #grad_output*torch.tensor(0.,))).view(-1)
        # print(x.size())
        # print((grad_base*(x>0).float()).cpu().detach().mean().view(-1))
        return grad_x, grad_alpha, grad_base, None 
"""

