import torch
import torch.nn as nn
from operations import *
from config import AugmentConfig

args = AugmentConfig()

class Cell(nn.Module):
    
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)
        
    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices
        
    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)

    def set_base(self):
        base_tmp = 0
        op_cnt = 0
        for op in self._ops:
            if op.op_type == 'zero':
                pass
            if op.op_type == 'fr':
                for seq in op.conv_1:
                    if hasattr(seq, 'tmp_base'):
                        base_tmp += seq.base.data
                        op_cnt += 1
                for seq in op.conv_2:
                    if hasattr(seq, 'tmp_base'):
                        base_tmp += seq.base.data
                        op_cnt += 1
            else:
                for seq in op.op:
                    if hasattr(seq, 'tmp_base'):
                        base_tmp += seq.base.data
                        op_cnt += 1
        with torch.no_grad():
            self.mean_base = base_tmp / op_cnt 
            if self.preprocess0.op_type == 'fr':
                for seq in self.preprocess0.conv_1:
                    if hasattr(seq, 'tmp_base'):
                        seq.base.data = torch.Tensor([(self.mean_base)]).cuda()
                for seq in self.preprocess0.conv_2:
                    if hasattr(seq, 'tmp_base'):
                        seq.base.data = torch.Tensor([(self.mean_base)]).cuda()
                    
            elif self.preprocess0.op_type == 'rcb':
                for seq in self.preprocess0.op:
                    if hasattr(seq, 'tmp_base'):
                        seq.base.data = torch.Tensor([(self.mean_base)]).cuda()
                
            for seq in self.preprocess1.op:
                if hasattr(seq, 'tmp_base'):
                    seq.base.data = torch.Tensor([(self.mean_base)]).cuda()
            
            for op in self._ops:
                if op.op_type == 'fr':
                    for seq in op.conv_1:
                        if hasattr(seq, 'tmp_base'):
                            seq.base.data = torch.Tensor([(self.mean_base)]).cuda()

                    for seq in op.conv_2:
                        if hasattr(seq, 'tmp_base'):
                            seq.base.data = torch.Tensor([(self.mean_base)]).cuda()

                else:
                    for seq in op.op:
                        if hasattr(seq, 'tmp_base'):
                            seq.base.data = torch.Tensor([(self.mean_base)]).cuda()
    
    def cell_energy(self):
        self.cell_e_add = 0
        self.cell_e_neuron = 0
        for i in range(len(self._ops)):
            op = self._ops[i]
            spike_data = op.spike_datas()
            op_flops_spike_rate = torch.sum(torch.stack([flops * spike_rate for flops, spike_rate in zip(spike_data[0], spike_data[1])]), dim=0)
            op_time_neuron = torch.sum(torch.stack(spike_data[2]), dim=0).sum()
            self.cell_e_add += 0.03 * op_flops_spike_rate
            self.cell_e_neuron += 0.26 * op_time_neuron
        return self.cell_e_add, self.cell_e_neuron
       
class NetworkCIFAR(nn.Module):
    
    def __init__(self, C, num_classes, layers, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        
        stem_multiplier = 3
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
            PACT_with_log_quantize()
        )
        
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        
    def forward(self, input):
        E_add = 0
        E_neuron = 0
        s0 = s1 = self.stem(input)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
            cell_e_add, cell_e_neuron = cell.cell_energy()
            E_add += cell_e_add
            E_neuron += cell_e_neuron
        self._total_spike_energy = E_add + E_neuron
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits, self._total_spike_energy
    
    