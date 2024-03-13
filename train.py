import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import genotypes
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter
from config import AugmentConfig
from torch.autograd import Variable
from model import NetworkCIFAR as Network

args = AugmentConfig()

device = torch.device("cuda")

writer = SummaryWriter(log_dir=os.path.join(args.path, "tb"))
writer.add_text('args', args.as_markdown(), 0)

logger = utils.get_logger(os.path.join(args.path, "{}.log".format(args.name)))
args.print_params(logger.info)
init_energy = 1.0


def main():
    logger.info("Logger is set - training start")
    
    if not torch.cuda.is_available():
        logger.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpus[0])
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)

    # get data with meta info
    n_classes, train_data, valid_data = utils.get_data(
        args.dataset, args.data_path, cutout_length=0)
        
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, n_classes, args.layers, args.auxiliary, genotype)
     
    #model = utils.load_checkpoint(model, args.load_dir, epoch=args.load_epoch)
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )
    
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, 
        shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=args.workers)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.learning_rate_min)
    best_acc = 0.0
    global init_energy
    for epoch in range(args.epochs):
        if epoch == args.basestep:
            model = base_mode_switch(model)
        scheduler.step()
        current_lr = scheduler.get_lr()[0]
        logger.info('Epoch: %d lr: %e', epoch, current_lr)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        train_acc, train_obj = train(train_queue, model, criterion, optimizer, epoch)
        logger.info('train_acc {:.4%}'.format(train_acc))

        valid_acc, valid_obj = infer(valid_queue, model, criterion, epoch)
        if valid_acc > best_acc:
            best_acc = valid_acc
        logger.info('valid_acc {:.4%}, best_acc {:.4%}'.format(valid_acc, best_acc))
        min_alpha, _ = print_minimum_alpha(model, 5)
        min_base, max_base, _ = print_min_max_base(model, 2, 0)
        logger.info('min_alpha %f', min_alpha)
        logger.info('min_base %.3f, max_base %.3f', min_base, max_base)
        if not os.path.exists(os.path.join(args.path, str(epoch))):
            os.mkdir(os.path.join(args.path, str(epoch)))
        utils.save_checkpoint(model, os.path.join(args.path, str(epoch)))

def print_minimum_alpha(model, min_alpha):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            min_alpha, model._modules[name] = print_minimum_alpha(
                                module, min_alpha)
        if (hasattr(module, "alpha") and hasattr(module, "base") ) :
            alpha_tmp = model._modules[name].alpha
            if min_alpha > alpha_tmp:
                min_alpha = alpha_tmp
    return min_alpha, model

def print_min_max_base(model, min_base, max_base):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            min_base, max_base, model._modules[name] = print_min_max_base(
                module, min_base, max_base)
        if (hasattr(module, "alpha") and hasattr(module, "base") ) :
            base_tmp = model._modules[name].base
            if min_base > base_tmp:
                min_base = base_tmp
            if max_base < base_tmp:
                max_base = base_tmp
    return min_base, max_base, model

def print_base(model, base, op_name='stem'):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            if hasattr(module, "op_type"):
                op_name = module.op_type
            base, model._modules[name] = print_base(module, base, op_name=op_name)
            
        if (hasattr(module, "alpha") and hasattr(module, "base") ) :
            base.append([op_name, model._modules[name].base]) #round(model._modules[name].base.data, 5)]) #model._modules[name].base.item()])
    return base, model

def base_mode_switch(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = base_mode_switch(module)
        if (hasattr(module, "alpha") and hasattr(module, "base") ) :
            model._modules[name].alpha.requires_grad = True
            model._modules[name].base.requires_grad = True
    return model
 
def train(train_queue, model, criterion, optimizer, epoch):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()
 
    global init_energy
    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        for cell in model.module.cells:
            cell.set_base()
        logits, logits_aux, spike_E = model(input)

        base, _ = print_base(model, [])
        spike_E = spike_E.mean()
        if epoch == 0 and step == 0:
            init_energy = spike_E
        loss = criterion(logits, target) #+ spike_E.detach() / init_energy.detach()
        for list in base:
            print(list)
            
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        losses.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)


        if step % args.print_freq == 0:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} Spike Energy {spike_E:.3f}  Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, args.epochs, step, len(train_queue) - 1, losses=losses, spike_E=spike_E.item(),
                    top1=top1, top5=top5))
            alpha, _ = print_minimum_alpha(model, 5)
            print('alpha', alpha)

    return top1.avg, losses.avg


def infer(valid_queue, model, criterion, epoch):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        logits, _, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        losses.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.print_freq == 0:
            logger.info("Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} Final Prec@1 {top1.avg:.4%}".format(epoch+1, args.epochs, step, len(valid_queue) - 1, losses=losses, top1=top1))

    return top1.avg, losses.avg


if __name__ == '__main__':
    main() 
