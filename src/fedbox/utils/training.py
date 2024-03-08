import random
from typing import Any, Union, Iterable, Optional, Mapping

import numpy as np
import torch
import torch.nn
import torch.cuda


class EarlyStopper:
    def __init__(self, higher_better: bool, patience: Optional[int] = None):
        if patience is not None and patience < 1:
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
        return self.patience is not None and self.worse_times >= self.patience

    def __getitem__(self, key: str):
        return self.dict[key]

    def __setitem__(self, key: str, value):
        self.dict[key] = value

    def __delitem__(self, key: str):
        del self.dict[key]

    def __contains__(self, key: str) -> bool:
        return key in self.dict


class MeanDict(Mapping[str, Any]):
    def __init__(self):
        self.__sum_count: dict[str, tuple[Any, int]] = {}

    def add(self, **values):
        for key, value in values.items():
            entry = self.__sum_count.get(key)
            if entry is None:
                self.__sum_count[key] = (value, 1)
            else:
                tot, cnt = entry
                self.__sum_count[key] = (tot + value), (cnt + 1)

    def __getitem__(self, key: str):
        tot, cnt = self.__sum_count[key]
        return tot / cnt

    def __delitem__(self, key: str):
        del self.__sum_count[key]

    def __contains__(self, key: str) -> bool:
        return key in self.__sum_count

    def clear(self):
        self.__sum_count.clear()

    def __len__(self) -> int:
        return len(self.__sum_count)

    def __iter__(self):
        return iter(self.__sum_count)

    def keys(self):
        return self.__sum_count.keys()


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
    freeze_module(module, requires_grad=True)
