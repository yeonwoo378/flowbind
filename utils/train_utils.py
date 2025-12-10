import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import matplotlib.pyplot as plt
import wandb


class WarmupConstantScheduler(_LRScheduler):
    """
    LR scheduler that warms up the learning rate for a given number of steps,
    then keeps it constant.
    """

    def __init__(self, optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # The 'self.last_epoch' is the current step counter
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            scale = self.last_epoch / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Learning rate is constant after warmup
            return self.base_lrs


def kl(mu_p, var_p, mu_q, var_q, eps: float = 1e-6):

    mu_q, var_q, mu_p, var_p = map(torch.as_tensor, (mu_q, var_q, mu_p, var_p))

    # Stabilise variances
    var_q = var_q.clamp_min(eps)
    var_p = var_p.clamp_min(eps)

    # Dimensionality (last axis)
    k = mu_q.shape[-1]

    log_ratio = torch.log(var_p) - torch.log(var_q)               # ... × D
    trace_term = var_q / var_p                                    # ... × D
    quad_term = (mu_p - mu_q).pow(2) / var_p                      # ... × D

    kl = 0.5 * (log_ratio + trace_term + quad_term - 1).mean()
    return kl


def get_z_star(feats, means, logvars, device, dropout=0.0):

    num_mods = len(feats)

    means_list = []
    vars_list = []
    for i in range(num_mods):
        p = torch.rand(1)
        if p.item() < dropout:  # dropout
            pass
        else:
            means_list.append(means[i].to(torch.float32))
            vars_list.append(torch.exp(logvars[i]).to(torch.float32))
    if len(means_list) == 0:
        means_list = [mean.to(torch.float32) for mean in means]
        vars_list = [torch.exp(logvar).to(torch.float32) for logvar in logvars]
    inverse_var = torch.zeros_like(vars_list[0])

    for i in range(len(vars_list)):
        inverse_var += 1.0 / vars_list[i]

    inverse_var_weighted_mean = torch.zeros_like(means_list[0])
    for i in range(len(means_list)):
        inverse_var_weighted_mean += means_list[i] / vars_list[i]

    z_mean = inverse_var_weighted_mean / inverse_var
    z_var = 1. / inverse_var

    loss = [kl(z_mean, z_var, means_list[i], vars_list[i])
            for i in range(len(means_list))]
    loss = sum(loss)

    return z_mean, z_var
