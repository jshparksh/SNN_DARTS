import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logger
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
    
    optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
               lr=args.arch_learning_rate, betas=(0.5, 0.999), 
               weight_decay=args.arch_weight_decay)

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))
    
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, 
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.workers)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=args.learning_rate_min)

    # training loop
    lr = args.learning_rate
    for epoch in range(args.epochs):
        scheduler.step()
        current_lr = scheduler.get_lr()[0]
        logger.info('Epoch: %d lr: %e', epoch, current_lr)
        
        # warmup
        if epoch < 5: # and args.batch_size > 256
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logger.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
        
        # genotype
        genotype = model.module.genotype()
        logger.info('genotype = %s', genotype)
        
        # alpha parameters
        arch_param = model.module.arch_parameters()
        logger.info(F.softmax(arch_param[0], dim=-1))
        logger.info(F.softmax(arch_param[1], dim=-1))
        
        # training
        train(train_queue, valid_queue, model, optimizer, optimizer_a, criterion, lr,epoch)
        
        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        #test_acc, test_obj = infer(test_queue, model, criterion)
        logger.info('Valid_acc %f', valid_acc)
        #logger.info('Test_acc %f', test_acc)

        #utils.save(model, os.path.join(args.save, 'weights.pt'))

def train(train_queue, valid_queue, model, optimizer, optimizer_a, criterion, lr, epoch):
    losses = utils.AvgrageMeter()
    arc_losses = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

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
        
        # after begin epoch, update alpha
        if epoch >= args.begin:
            optimizer_a.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)
            loss_a.backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            optimizer_a.step()
        #architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        
        optimizer.step()


        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        losses.update(loss.data.item(), n)
        arc_losses.update(loss_a.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logger.info(
                    "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} Arc_Loss {arc_losses.avg:.3f}"
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, args.epochs, step, len(train_queue) - 1, losses=losses, arc_losses=arc_losses, 
                        top1=top1, top5=top5))

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, args.epochs, top1.avg))



def infer(valid_queue, model, epoch, criterion):
    losses = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
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

        if step % args.report_freq == 0:
            logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, args.epochs, step, len(valid_queue)-1, losses=losses,
                        top1=top1, top5=top5))

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, args.epochs, top1.avg))
    return top1.avg, losses.avg


if __name__ == '__main__':
    main() 

