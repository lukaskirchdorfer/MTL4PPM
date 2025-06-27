import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.weighting.abstract_weighting import AbsWeighting


class UW_SO(AbsWeighting):
    r"""(Soft) Optimal Uncertainty Weighting
    This method is proposed in 'Analytical Uncertainty-based Loss Weighting for Multi-Task Learning (GCPR 2024) <https://arxiv.org/abs/2408.07985>'

    Args:
        T (float, default=1.0): The temperature parameter for the softmax function.
        use_softmax (bool, default=True): Whether to use the softmax function to normalize the weights.
    """

    def __init__(self):
        super(UW_SO, self).__init__()

    def backward(self, losses, **kwargs):
        T = kwargs["T"]
        use_softmax = kwargs["use_softmax"]
        E = 1e-8
        losses_detached = losses.detach()

        if use_softmax == False:
            batch_weight = 1 / (losses_detached + E)
        else:
            batch_weight = F.softmax((1 / (losses_detached + E)) / T, dim=-1)
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode='autograd') # [task_num, grad_dim]
        loss = (batch_weight * losses).sum()
        loss.backward()
        return batch_weight.detach().cpu().numpy(), grads