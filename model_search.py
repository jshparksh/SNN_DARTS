import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
from quantization import *
from config import SearchConfig

args = SearchConfig()

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._op_energy = 0
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            self._ops.append(op)

    def forward(self, x, weights):
        # fix me : can get flops, spike_rate, time_neuron from each operation in list format
        # iteration to get each operation's flops, spike_rate, time_neuron, and calculate energy and return as summed value
        self.ofms = [op(x) for op in self._ops]
        
        return sum(w * ofm for w, ofm in zip(weights, self.ofms))
    
    def op_energy(self, alphas):
        # Using alpha (operation probability) to calculate energy of MixedOP
        # Diff value for backward will be energy value
        mixed_add = 0
        mixed_neuron = 0
        
        for op_idx in range(len(self._ops)):
            spike_data = self._ops[op_idx].spike_datas() # spike_data = [flops, spike_rate, time_neuron]
            op_flops_spike_rate = torch.sum(torch.stack([flops * spike_rate for flops, spike_rate in zip(spike_data[0], spike_data[1])]), dim=0)
            op_time_neuron = torch.sum(torch.stack(spike_data[2]), dim=0).sum()
            mixed_add += 0.03 * op_flops_spike_rate * alphas[op_idx]
            mixed_neuron += 0.26 * op_time_neuron * alphas[op_idx]
        self._op_e_add = mixed_add
        self._op_e_neuron = mixed_neuron
        
        return self._op_e_add, self._op_e_neuron
        

class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, base=math.sqrt(2), time_step=args.timestep, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        
        return torch.cat(states[-self._multiplier:], dim=1)

    def cell_energy(self, weights):
        self.cell_e_add = 0
        self.cell_e_neuron = 0
        for i in range(len(self._ops)):
            op = self._ops[i]
            op_alpha = weights[i] # op_alpha len = # of operations
            op_e_add, op_e_neuron = op.op_energy(op_alpha)
            self.cell_e_add += op_e_add
            self.cell_e_neuron += op_e_neuron
        return self.cell_e_add, self.cell_e_neuron
    
class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_curr),
            PACT()
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        
        self.cells = nn.ModuleList()
        self._total_spike_energy = torch.Tensor([1]).cuda()
        
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
                
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()
        
    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, spike_bool=False):
        E_add = 0
        E_neuron = 0
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        if spike_bool == True:
            for i, cell in enumerate(self.cells):
                if cell.reduction:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
                cell_e_add, cell_e_neuron = cell.cell_energy(weights)
                E_add += cell_e_add
                E_neuron += cell_e_neuron
            self._total_spike_energy = E_add + E_neuron
            return logits, self._total_spike_energy
        return logits
    
    def spike_energy(self):
        return self._total_spike_energy

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = nn.Parameter(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = nn.Parameter(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                
                #edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene
        
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
    
    
        
            
            
