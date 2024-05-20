""" Config class for search/augment """
import argparse
import os
import genotypes as gt
from functools import partial
import torch


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class SearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', type=str, default='CIFAR10', help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--load', type=bool, default=False, help='load pretrained model')
        parser.add_argument('--save_dir', type=str, default='./searchs/', help='path to save results')
        parser.add_argument('--load_dir', type=str, default='./searchs/0209_spikeloss0.2/', help='path to save results')
        parser.add_argument('--load_epoch', type=str, default=0, help='load pretrained model from specific epoch')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        parser.add_argument('--learning_rate', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--learning_rate_base', type=float, default=1, help='lr for base')
        parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
        parser.add_argument('--learning_rate_min_base', type=float, default=0.05, help='min learning rate for base')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay for weights')
        parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--gpus', default='0,1,2,3', help='gpu device ids separated by comma. '
                                                                '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=100, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--layers', type=int, default=8, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--arch_learning_rate', type=float, default=6e-3, help='learning rate for arch encoding')
        parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
        parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
        parser.add_argument('--begin', type=int, default=15, help='batch size') # imagenet -> 35
        parser.add_argument('--spike_step', type=int, default=50, help='training with spike loss')
        parser.add_argument('--spike_bool', type=bool, default=False, help='to check status of training with spike loss')
        parser.add_argument('--timestep', type=int, default=16, help='timestep for logarithmic spike')
        parser.add_argument('--warmup', type=int, default=5, help='alpha, base requires_grad switch into true')
        parser.add_argument('--alpha_base_fix_epoch', type=int, default=50, help='alpha & base requires_grad switch into false')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = os.path.join('/data/', self.dataset)
        self.path = os.path.join('searchs', self.name)
        self.gpus = parse_gpus(self.gpus)


class AugmentConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Augment config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', type=str, default='CIFAR10', help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--arch', type=str, default='SNNDARTS', help='Cell genotype')
        parser.add_argument('--save_dir', type=str, default='./searchs/', help='path to save results')
        parser.add_argument('--load_dir', type=str, default='./augments/0226_nolog/', help='path to save results')
        parser.add_argument('--load_epoch', type=str, default='0', help='load pretrained model from specific epoch')
        parser.add_argument('--batch_size', type=int, default=512, help='batch size')
        parser.add_argument('--learning_rate', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--learning_rate_base', type=float, default=0.1, help='lr for base')
        parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
        parser.add_argument('--learning_rate_min_base', type=float, default=0.005, help='min learning rate for base')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
        parser.add_argument('--print_freq', type=int, default=1, help='print frequency')
        parser.add_argument('--gpus', default='0,1,2,3', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=600, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=32)
        parser.add_argument('--layers', type=int, default=16, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
        parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='auxiliary loss weight')
        parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
        parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path prob')
        parser.add_argument('--timestep', type=int, default=4, help='timestep for logarithmic spike')
        parser.add_argument('--warmup', type=int, default=5, help='alpha, base requires_grad switch into true')
        parser.add_argument('--alpha_base_fix_epoch', type=int, default=100, help='alpha & base requires_grad switch into false')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './data/'
        self.path = os.path.join('augments', self.name)
        #self.genotype = gt.from_str(self.genotype)
        self.gpus = parse_gpus(self.gpus)
        
