import torch
import torch.nn as nn
import numpy as np


class Architect(object):
    
    def __init__(self, model, criterion, args):
        self.model = model
        self.criterion = criterion
        self.grad_clip = args.grad_clip
        self.max_E = None
        self.optimizer = torch.optim.Adam(model.module.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999), 
                                          weight_decay=args.arch_weight_decay)
        
    def step(self, input_valid, target_valid, spike_bool=False):
        self.optimizer.zero_grad()
        self._backward_step(input_valid, target_valid, spike_bool)
        nn.utils.clip_grad_norm_(self.model.module.arch_parameters(), self.grad_clip)
        self.optimizer.step()
        
    def _backward_step(self, input_valid, target_valid, spike_bool):
        if spike_bool == False:
            loss = self.criterion(self.model(input_valid), target_valid)
            self.loss = loss
        else:
            loss = self._compute_loss(self.model, input_valid, target_valid, spike_bool)
        loss.backward()
    
    def _compute_loss(self, model, input_valid, target_valid, spike_bool):
        logit, spike_E = model(input_valid, spike_bool)
        loss = self.criterion(logit, target_valid)
        spike_E = spike_E.mean()
        # max_E at initial spike loss calculation for normalization
        if self.max_E == None: #epoch == self.spike_step:
            self.max_E = spike_E
        spike_loss = spike_E/self.max_E.detach() #detach() for preventing double backpropagation
        lmd1 = 1
        lmd2 = 1/5
        # for logging
        self.loss = loss
        self.spike_loss = spike_loss
        new_loss = lmd1*loss + lmd2*spike_loss
        
        # for logging
        self.loss = loss
        self.spike_E = spike_E
        self.spike_loss = spike_loss
        self.arc_loss = new_loss
        
        return new_loss