import pandas as pd
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

class Writer:
    """Computes and stores the average and current value"""
    
    def __init__(self, name, fmt=':.4e', summary_type=Summary.AVERAGE):
        
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
    
    def save(self, model_name):
        
        self.file = conf.out_history/f"Model_{model_name}_LOSS.txt"
        print(self.summary())
        with open(self.file, "a") as f:
            f.write(json.dumps(self.summary()))
            f.write("\n")
            
    def plot_metrics(self, file_path=None,):
        """Create a graph with the metrics caputred in the file"""
        
        metrics = ["Train loss", "Validation loss", "Accuracy"]
        results = {}
        
        file_src = file_path or Path(self.file)

        with open(file_src, "r") as file:
            for line in file.readlines():

                epoch = line.split()[1]
                name = next(iter([metric for metric in metrics if metric in line]))
                value = line.split()[-2]

                if epoch in results:
                    results[epoch].append(value)
                else:
                    results[epoch] = [value]
                    
        results_df = pd.DataFrame.from_dict(results, orient="index", columns=metrics,)
        results_df = results_df.astype(float)
        results_df.iloc[:, :2].plot()
        results_df.iloc[:, 2:].plot()
        
        return results_df