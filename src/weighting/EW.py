import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.weighting.abstract_weighting import AbsWeighting

class EW(AbsWeighting):
    r"""Equal Weighting (EW).

    The loss weight for each task is always ``1 / T`` in every iteration, where ``T`` denotes the number of tasks.

    """
    def __init__(self):
        super(EW, self).__init__()
        
    def backward(self, losses, **kwargs):
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode='autograd') # [task_num, grad_dim]
        loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
        loss.backward()
        return np.ones(len(losses)), grads