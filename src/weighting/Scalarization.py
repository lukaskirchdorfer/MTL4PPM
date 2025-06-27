import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.weighting.abstract_weighting import AbsWeighting


class Scalarization(AbsWeighting):
    r"""Scalarization.

    Grid Search across static task weights based on Xin et al. (Google Paper)

    """

    def __init__(self):
        super(Scalarization, self).__init__()

    def backward(self, losses, **kwargs):
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode='autograd') # [task_num, grad_dim]
        scalar_weights = kwargs["scalar_weights"]
        if scalar_weights is None:
            raise ValueError("scalar_weights must be provided")
        weights = torch.tensor(scalar_weights).to(self.device)
        loss = torch.mul(losses, weights).sum()
        loss.backward()
        return weights.detach().cpu().numpy(), grads