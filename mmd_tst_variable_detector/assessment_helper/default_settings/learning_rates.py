import functools
import torch


lr_scheduler = functools.partial(torch.optim.lr_scheduler.ReduceLROnPlateau,
                                 mode='min',
                                 min_lr=0.001)
initial_lr_rate: float = 0.01
