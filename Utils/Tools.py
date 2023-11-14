import torch
from torch import nn
import torch.nn.functional as F
import functools
import random
import numpy as np
import datetime
import io


def fixSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multiGPUs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr): return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class MyTqdm:
    def __init__(self, obj, print_step=150, total=None):
        self.obj = iter(obj)
        self.len = len(obj) if total is None else total
        self.print_step = print_step
        self.idx = 0
        self.msg = 'None'

    def __len__(self):
        return self.len

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.idx == 0: self.start = datetime.datetime.now()
        out = next(self.obj)
        self.idx += 1
        if self.idx % self.print_step == 0 or self.idx == len(self)-1:
            delta = datetime.datetime.now() - self.start
            avg_sec_per_iter = delta.total_seconds() / float(self.idx)

            total_time_pred = datetime.timedelta(seconds=round(avg_sec_per_iter * len(self)))
            delta = datetime.timedelta(seconds=round(delta.total_seconds()))
            if avg_sec_per_iter > 1:
                s = '[%d/%d]  [%.2f s/it]  [%s]  [%s /epoch]'%(self.idx, len(self), avg_sec_per_iter, str(delta), str(total_time_pred))
            else:
                s = '[%d/%d]  [%.2f it/s]  [%s]  [%s /epoch]'%(self.idx, len(self), 1/avg_sec_per_iter, str(delta), str(total_time_pred))
            print (s)
            self.msg = s

        return out

    def getMessage(self):
        return self.msg