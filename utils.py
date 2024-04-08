""" Utilities """
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import preproc


def get_data(dataset, data_path, cutout_length):
    """ Get torchvision dataset """
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
    elif dataset == 'fashionmnist':
        dset_cls = dset.FashionMNIST
        n_classes = 10
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
    trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)
    val_data = dset_cls(root=data_path, train=False, download=True, transform=val_transform)

    ret = [n_classes, trn_data, val_data]

    return ret


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)

def load_checkpoint(model, load_dir, epoch=None, is_best=False):
    if is_best:
        ckpt = os.path.join(load_dir, "best.pth.tar")
    elif epoch is not None:
        ckpt = os.path.join(load_dir, epoch, "checkpoint.pth.tar".format(epoch))
    else:
        ckpt = os.path.join(load_dir, "checkpoint.pth.tar")
    model.load_state_dict(torch.load(ckpt).module.state_dict(), strict=False) #.module.state_dict()
    return model

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x

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

def print_base_grad(model, tmp_base, op_name='stem'):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            if hasattr(module, "op_type"):
                op_name = module.op_type
            tmp_base, model._modules[name] = print_base_grad(module, tmp_base, op_name=op_name)
        if (hasattr(module, "alpha") and hasattr(module, "base") ) :
            tmp_base.append([op_name, round(model._modules[name].tmp_base.grad.item(), 5)])
    return tmp_base, model

def print_base(model, base, op_name='stem'):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            if hasattr(module, "op_type"):
                op_name = module.op_type
            base, model._modules[name] = print_base(module, base, op_name=op_name)
            
        if (hasattr(module, "alpha") and hasattr(module, "base") ) :
            base.append([op_name, round(model._modules[name].base.item(), 5)]) #round(model._modules[name].base.data, 5)]) #model._modules[name].base.item()])
    return base, model

def update_base(model, step):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = update_base(module, step)
        if (hasattr(module, "alpha") and hasattr(module, "base") ):
            model._modules[name].base += model._modules[name].tmp_base.data / step
            model._modules[name].tmp_base.data *= 0
    return model

# edit here
def param_mode_switch(model, grad_bool=True):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = param_mode_switch(module, grad_bool)
        if (hasattr(module, "alpha") and hasattr(module, "base") ) :
            model._modules[name].alpha.requires_grad = grad_bool
            model._modules[name].tmp_base.requires_grad = grad_bool
    return model

def base_mode_switch(model, grad_bool=True):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = base_mode_switch(module, grad_bool)
        if (hasattr(module, "alpha") and hasattr(module, "base") ) :
            model._modules[name].tmp_base.requires_grad = grad_bool
    return model

def split_params(model):
    tmp_base_params = []
    base_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'tmp_base' in name:
            tmp_base_params.append(param)
        elif 'base' in name:
            base_params.append(param)
        else:
            other_params.append(param)
    return tmp_base_params, other_params