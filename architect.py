import torch
import torch.nn as nn
import numpy as np


class Architect(object):
    
    def __init__(self, model, criterion, args):
        self.model = model
        self.criterion = criterion
        self.spike_step = args.spike_step
        self.max_E = 1
        self.optimizer = torch.optim.Adam(model.module.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999), 
                                          weight_decay=args.arch_weight_decay)
        
    def step(self, input_valid, target_valid, epoch):
        self.optimizer.zero_grad()
        self._backward_step(input_valid, target_valid, epoch)
        self.optimizer.step()
        
    def _backward_step(self, input_valid, target_valid, epoch):
        if epoch < self.spike_step:
            loss = self.criterion(self.model(input_valid), target_valid)
        else:
            loss = self._compute_loss(self.model(input_valid), target_valid, epoch)
        loss.backward()
    
    def _compute_loss(self, input_valid, target_valid, epoch):
        loss = self.criterion(input_valid, target_valid)
        spike_E = self.model._spike_energy()
        # max_E at initial spike loss calculation for normalization
        if epoch == self.spike_step:
            self.max_E = spike_E
        spike_loss = spike_E/self.max_E
        lmd1 = 1/2
        lmd2 = 1/2
        new_loss = lmd1*loss + lmd2*spike_loss
        
        # for logging
        self.loss = loss
        self.spike_loss = spike_loss
        self.arc_loss = new_loss
        
        return new_loss