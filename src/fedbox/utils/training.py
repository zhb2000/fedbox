import random
from typing import Any, Union, Iterable

import numpy as np
import torch
import torch.nn
import torch.cuda

__all__ = [
    'EarlyStopper',
    'set_seed',
    'freeze_module',
    'unfreeze_module'
]


class EarlyStopper:
    def __init__(self, higher_better: bool, patience: int = 1):
        if patience < 1:
            raise ValueError("'patience' must be at least 1")
        self.higher_better = higher_better
        self.best_metric: float = -np.inf if higher_better else np.inf
        self.patience = patience
        self.worse_times = 0
        self.dict: dict[str, Any] = {}

    def is_better(self, metric: float) -> bool:
        return (
            metric > self.best_metric if self.higher_better 
            else metric < self.best_metric
        )

    def update(self, metric: float, **kwargs):
        """
        :param metric: new metric value
        :param kwargs: some other information with this metric
        :return: whether self is updated (`metric` becomes the new best metric).
        """
        if self.is_better(metric):
            self.best_metric = metric
            self.worse_times = 0
            for key, value in kwargs.items():
                self.dict[key] = value
            return True
        else:
            self.worse_times += 1
            return False

    def reach_stop(self) -> bool:
        return self.worse_times >= self.patience

    def __getitem__(self, key: str):
        return self.dict[key]

    def __setitem__(self, key: str, value):
        self.dict[key] = value

    def __delitem__(self, key: str):
        del self.dict[key]

    def __contains__(self, key: str) -> bool:
        return key in self.dict


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_module(
    module: Union[torch.nn.Module, Iterable[torch.Tensor]],
    requires_grad: bool = False
):
    params = module.parameters() if isinstance(module, torch.nn.Module) else module
    for p in params:
        p.requires_grad = requires_grad


def unfreeze_module(module: Union[torch.nn.Module, Iterable[torch.Tensor]]):
    freeze_module(module, False)
