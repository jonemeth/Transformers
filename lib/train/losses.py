from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class Loss(nn.Module, ABC):
    @abstractmethod
    def prepare_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented


class CrossEntropy(Loss, nn.CrossEntropyLoss):
    def prepare_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, y


class CutMix(Loss):
    def __init__(self, loss, alpha=1.0):
        super().__init__()
        self.loss = loss
        self.alpha = alpha
        self.rng = np.random.default_rng()

        self.index = None
        self.lamb = None

    def prepare_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.index = torch.randperm(x.size(0)).to(x.device)

        self.lamb = self.rng.beta(self.alpha, self.alpha)
        y1, x1, y2, x2 = self._cut_bounding_box(x.shape[-2:], self.lamb)
        x[:, :, y1:y2, x1:x2] = x[self.index, :, y1:y2, x1:x2]

        # adjust lambda to exactly match pixel ratio
        area = x.size(2) * x.size(3)
        self.lamb = 1. - (x2 - x1) * (y2 - y1) / area
        return x, y

    def _cut_bounding_box(self, shape, lamb):
        cut_size_2 = 0.5 * np.sqrt(1. - lamb)
        center_yx = self.rng.random(2)

        y1x1 = (np.clip(center_yx - cut_size_2, 0., 1.) * shape).astype(int)
        y2x2 = (np.clip(center_yx + cut_size_2, 0., 1.) * shape).astype(int)
        return np.concatenate((y1x1, y2x2))

    def forward(self, pred, target):
        orig_reduction = self.loss.reduction
        self.loss.reduction = 'none'
        batch_loss = self.lamb * self.loss(pred, target) + (1. - self.lamb) * self.loss(pred, target[self.index])
        self.loss.reduction = orig_reduction
        return reduce_loss(batch_loss, orig_reduction)
