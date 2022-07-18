from pathlib import Path
import torch.distributed as dist
import torch
from enum import Enum
import time
from config import conf
import json

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name, epoch, model_id, suff="", fmt=':.4e', summary_type=Summary.AVERAGE):
        
        self.suff = suff
        self.model_id = model_id
        self.epoch = epoch
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.current_batch = 0

    def update(self, val, epoch, n=1, ):
        
        self.epoch = epoch
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count])
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = 'Epoch: {epoch} {name} {avg:.3f} '
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)
    
    def save(self):
        
        file = conf.out_history.with_name(f"Model_{self.model_id}_{self.suff}.txt")
        with open(file, "a") as f:
            f.write(json.dumps(self.summary()))
            f.write("\n")
        
        