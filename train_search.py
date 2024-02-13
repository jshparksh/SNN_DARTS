import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import copy

from tensorboardX import SummaryWriter
from config import SearchConfig
from torch.autograd import Variable
from model_search import Network
from architect import Architect

args = SearchConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(args.path, "tb"))
writer.add_text('args', args.as_markdown(), 0)

logger = utils.get_logger(os.path.join(args.path, "{}.log".format(args.name)))
args.print_params(logger.info)

def main():
    logger.info("Logger is set - training start")
    
    if not torch.cuda.is_available():
        logger.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpus[0])
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled=True
    
    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        args.dataset, args.data_path, cutout_length=0, validation=False)
    
    criterion = nn.CrossEntropyLoss().cuda()
    
    model = Network(args.init_channels, n_classes, args.layers, criterion)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    
    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))
    
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, 
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:n_train]),
        pin_memory=True, num_workers=args.workers)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=args.learning_rate_min)

    architect = Architect(model, criterion, args)
    
    # training loop
    lr = args.learning_rate
    for epoch in range(args.epochs):
        scheduler.step()
        current_lr = scheduler.get_lr()[0]
        logger.info('Epoch: %d lr: %e', epoch, current_lr)
        
        """# warmup
        if epoch < 5: # and args.batch_size > 256
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logger.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)"""
        
        # genotype
        genotype = model.module.genotype()
        logger.info('genotype = %s', genotype)
        
        # alpha parameters
        arch_param = model.module.arch_parameters()
        logger.info(F.softmax(arch_param[0], dim=-1))
        logger.info(F.softmax(arch_param[1], dim=-1))
        
        if epoch == args.begin:
            logger.info('Begin architect search')
            
        if epoch == args.spike_step:
            logger.info('Begin spike-aware search')
            
        # training
        train(train_queue, valid_queue, model, architect, optimizer, criterion, lr, epoch)
        
        if epoch >= args.spike_step:
            # validation
            valid_acc, valid_obj = infer(valid_queue, model, epoch, criterion)
            logger.info('Valid_acc %f', valid_acc)
        min_alpha, _ = print_minimum_alpha(model, 5)
        logger.info('min_alpha %f', min_alpha)
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

def train(train_queue, valid_queue, model, architect, optimizer, criterion, lr, epoch):
    losses = utils.AverageMeter()
    arc_losses = utils.AverageMeter()
    spike_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)
        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        # after begin epoch, update alpha
        if epoch >= args.spike_step:
            architect.step(input_search, target_search, spike_bool=True)
            arc_losses.update(architect.loss.item(), n)
            spike_losses.update(architect.spike_loss.item(), n)
        elif epoch >= args.begin:
            architect.step(input_search, target_search, spike_bool=False)
            arc_losses.update(architect.loss.item(), n)
                
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        losses.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.print_freq == 0:
            if epoch >= args.begin:
                if epoch < args.spike_step:
                    logger.info(
                        "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} Arc_Loss {arc_losses.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                            epoch + 1, args.epochs, step, len(train_queue) - 1, losses=losses, arc_losses=arc_losses,
                            top1=top1, top5=top5))
                else:
                    logger.info(
                        "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} Arc_Loss {arc_losses.avg:.3f} Spike_Loss {spike_losses.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                            epoch + 1, args.epochs, step, len(train_queue) - 1, losses=losses, arc_losses=arc_losses, spike_losses=spike_losses,
                            top1=top1, top5=top5))
                    #logger.info("Alpha normal gradient: {}".format(architect.normal_grad))
                    #logger.info("Alpha reduce gradient: {}".format(architect.reduce_grad))
                    
            else:
                logger.info(
                    "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, args.epochs, step, len(train_queue) - 1, losses=losses, 
                        top1=top1, top5=top5))
        if step == 200:
            break
    logger.info("Train: [{:2d}/{}] Final Prec {:.4%}".format(epoch+1, args.epochs, top1.avg))
    if epoch >= args.spike_step:
        logger.info("Train: [{:2d}/{}] Spike Energy {:.4f}".format(epoch+1, args.epochs, architect.spike_E.item()))
    if epoch == args.spike_step:
        logger.info("Maximum spike energy: {:.4f}".format(architect.max_E.item()))



def infer(valid_queue, model, epoch, criterion):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        losses.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.print_freq == 0:
            logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, args.epochs, step, len(valid_queue)-1, losses=losses,
                        top1=top1, top5=top5))
        if step == 50:
            break
    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, args.epochs, top1.avg))
    return top1.avg, losses.avg


if __name__ == '__main__':
    main() 

