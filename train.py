import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
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

def main():
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
  input_size, input_channels, n_classes, train_data = utils.get_data(
    args.dataset, args.data_path, cutout_length=0, validation=False)
    
  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, n_classes, args.layers, args.auxiliary, genotype)
  model = torch.nn.DataParallel(model)
  model = model.cuda()

  logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

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
  
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  best_acc = 0.0
  for epoch in range(args.epochs):
    scheduler.step()
    current_lr = scheduler.get_lr()[0]
    logger.info('Epoch: %d lr: %e', epoch, current_lr)
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer, epoch)
    logger.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion, epoch)
    if valid_acc > best_acc:
        best_acc = valid_acc
    logger.info('valid_acc %f, best_acc %f', valid_acc, best_acc)

    if not os.path.exists(os.path.join(args.path, str(epoch))):
      os.mkdir(os.path.join(args.path, str(epoch)))
    utils.save_checkpoint(model, os.path.join(args.path, str(epoch)))


def train(train_queue, model, criterion, optimizer, epoch):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    optimizer.zero_grad()
    logits, logits_aux, spike_E = model(input)
    spike_E = spike_E.mean()
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.print_freq == 0:
      logger.info(
        "Train: [{:2d}/{}] Step {:03d}/{:03d} Spike Energy {:.3f} Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, args.epochs, step, len(train_queue) - 1, spike_E.item(), 
                        top1=top1, top5=top5))

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, epoch):
  objs = utils.AverageMeter()
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
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.print_freq == 0:
      logger.info("Valid: [{:2d}/{}] Step {:03d}/{:03d} Final Prec@1 {:.4%}".format(epoch+1, args.epochs, step, len(valid_queue) - 1, top1.avg))

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 
