import os
import sys
import numpy as np
import torch
import utils
import genotypes
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter
from config import AugmentConfigImageNet
from torch.autograd import Variable
#from cnn_test import SimpleCNN as Network
from model import NetworkImageNet as Network

args = AugmentConfigImageNet()

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
    model = Network(args.init_channels, n_classes, args.layers, genotype)
    if args.load == True:
        model = utils.load_checkpoint(model, args.load_dir, epoch=args.load_epoch)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=args.gpus)
    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss().cuda()
    
    # split model parameters into two groups
    alpha_params, base_params, model_params = utils.split_params(model)
    
    # sparate optimizer for alpha, base and model
    optimizer = torch.optim.SGD(
        model_params,
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )
    optimizer_alpha = torch.optim.SGD(
        alpha_params,
        args.learning_rate_alpha
    )
    optimizer_base = torch.optim.SGD(
        base_params,
        args.learning_rate_base
        )
    
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, 
        shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,
        shuffle=False, pin_memory=True, num_workers=args.workers)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scheduler_alpha = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_alpha, args.epochs)
    scheduler_base = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_base, args.epochs)
    best_acc = 0.0
    global init_energy
    for epoch in range(args.epochs):
        if epoch == args.warmup:
            model = utils.param_mode_switch(model)
            
        # if epoch == args.alpha_base_fix_epoch:
        #     model = utils.base_mode_switch(model, grad_bool=False)
        
        scheduler.step()
        current_lr = scheduler.get_lr()[0]
        #logger.info('Epoch: %d lr: %e', epoch, current_lr)
        scheduler_alpha.step()
        current_alpha_lr = scheduler_alpha.get_lr()[0]
        scheduler_base.step()
        current_base_lr = scheduler_base.get_lr()[0]
        logger.info('Epoch: %d lr: %e alphalr: %e baselr: %e', epoch, current_lr, current_alpha_lr, current_base_lr)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        train_acc, train_obj = train(train_queue, model, model_params, criterion, optimizer, optimizer_alpha, optimizer_base, epoch)
        logger.info('train_acc {:.4%}'.format(train_acc))

        valid_acc, valid_obj = infer(valid_queue, model, criterion, epoch)
        if valid_acc > best_acc:
            best_acc = valid_acc
        logger.info('valid_acc {:.4%}, best_acc {:.4%}'.format(valid_acc, best_acc))
        min_alpha, _ = utils.print_minimum_alpha(model, 1e6)
        min_base, max_base, _ = utils.print_min_max_base(model, 1e6, 0)
        logger.info('min_alpha %f', min_alpha)
        logger.info('min_base %.3f, max_base %.3f', min_base, max_base)
        # if not os.path.exists(os.path.join(args.path, str(epoch))):
        #     os.mkdir(os.path.join(args.path, str(epoch)))
        # utils.save_checkpoint(model, os.path.join(args.path, str(epoch)))
 
def train(train_queue, model, model_params, criterion, optimizer, optimizer_alpha, optimizer_base, epoch):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()
 
    global init_energy
    update_step = 1
    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        optimizer_alpha.zero_grad()
        optimizer_base.zero_grad()
        logits, spike_E = model(input)
        
        spike_E = spike_E.mean()
        if epoch == 0 and step == 0:
            init_energy = spike_E
        
        # edit here
        # fix base for each cell
        for cell in model.module.cells: #module.cells:
            cell.set_alpha_base()
            
        """
        if epoch >= args.warmup:
            loss = criterion(logits, target) + spike_E / init_energy.detach()
        else:
            loss = criterion(logits, target)
        """
        loss = criterion(logits, target)

        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model_params, args.grad_clip)
        optimizer_base.step()
        optimizer_alpha.step()
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
        if epoch >= args.warmup:
            if step % args.print_freq == 0:
                min_alpha, _ = utils.print_minimum_alpha(model, 1e6)
                print('min_alpha', min_alpha, 'step', step)
                alpha, _ = utils.print_alpha(model, [])
                alpha_grad, _ = utils.print_alpha_grad(model, [])
                base, tmp_base, _ = utils.print_base_tmpbase(model, [], [])
                base_grad, _ = utils.print_base_grad(model, [])
                print('-----------------alpha-----------------')
                for i in range(len(base)):
                    print(alpha[i][0], alpha[i][1], alpha_grad[i][1])
                print('-----------------base-----------------')
                for i in range(len(base)):
                    print(base[i][0], base[i][1], tmp_base[i][1], base_grad[i][1])
            
            if step != 0 and step % update_step == 0:
                utils.update_base(model, update_step)
    return top1.avg, losses.avg


def infer(valid_queue, model, criterion, epoch):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        logits, _ = model(input)
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
