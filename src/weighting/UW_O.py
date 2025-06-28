import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.weighting.abstract_weighting import AbsWeighting


class UW_O(AbsWeighting):
    r"""Optimal Uncertainty Weighting
    This method is proposed in 'Analytical Uncertainty-based Loss Weighting for Multi-Task Learning (GCPR 2024) <https://arxiv.org/abs/2408.07985>'

    """

    def __init__(self):
        super(UW_O, self).__init__()

    def backward(self, losses, **kwargs):
        E = 1e-8
        losses_detached = losses.detach()

        batch_weight = 1 / (losses_detached + E)

        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode='autograd') # [task_num, grad_dim]
        loss = (batch_weight * losses).sum()
        loss.backward()
        return batch_weight.detach().cpu().numpy(), grads