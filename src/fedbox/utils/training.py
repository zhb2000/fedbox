import random
from typing import Any, Union, Iterable, Optional, Mapping

import numpy as np
import torch
import torch.nn
import torch.cuda


class Recorder:
    """
    A utility class for performing early stopping in training processes. The `Recorder` tracks a specific 
    metric during training and determines whether to continue training based on the improvement of the metric. 
    It also supports patience, allowing training to continue for a specified number of epochs without improvement.

    Example:

        >>> from fedbox.utils.training import Recorder
        >>> stopper = Recorder(higher_better=True, patience=10)
        >>> for epoch in range(epochs):
        ...     train(...)
        ...     f1, acc = validate(...)
        ...     # Use 'f1' as the early stopping metric, and record 'acc' and 'epoch'
        ...     is_best = stopper.update(f1, acc=acc, epoch=epoch)
        ...     print(f'epoch {epoch}, is best: {is_best}, f1: {f1:.4f}, acc: {acc:.4f}')
        ...     if stopper.reach_stop():
        ...         break
        >>> # Print final result
        >>> print(f'best f1: {stopper.best_metric}, best epoch: {stopper["epoch"]}, acc: {stopper["acc"]}')
    """
    def __init__(self, higher_better: bool, patience: Optional[int] = None):
        """
        Initializes the Recorder.

        Args:
            higher_better (bool): Indicates whether a higher metric value is better (e.g., accuracy, F1-score).
            patience (Optional[int]): Number of epochs to wait for an improvement before stopping. 
                If `None`, early stopping is disabled.

        Raises:
            ValueError: If `patience` is provided and is less than 1.
        """
        if patience is not None and patience < 1:
            raise ValueError("'patience' must be at least 1")
        self.higher_better = higher_better
        """Indicates whether a higher metric value is better (e.g., accuracy, F1-score)."""

        self.best_metric: float = -np.inf if higher_better else np.inf
        """The best metric value observed so far."""

        self.patience = patience
        """Number of epochs to wait for an improvement before stopping. 
            If `None`, early stopping is disabled."""
        self.worse_times = 0
        """The number of consecutive epochs without improvement."""

        self.best_info: dict[str, Any] = {}
        """A dictionary to store additional information (e.g., epoch, accuracy) 
            associated with the best metric."""

    def is_better(self, metric: float) -> bool:
        """
        Checks if the new metric value is better than the current best metric.

        Args:
            metric (float): The new metric value.

        Returns:
            bool: True if the new metric is better, False otherwise.
        """
        return (
            metric > self.best_metric if self.higher_better 
            else metric < self.best_metric
        )

    def update(self, metric: float, **kwargs):
        """
        Updates the best metric if the new metric is better, and resets patience if improvement is observed.

        Args:
            metric (float): The new metric value.
            kwargs (dict): Additional information to store with the new best metric (e.g., epoch, accuracy).

        Returns:
            bool: True if the best metric is updated, False otherwise.
        """
        if self.is_better(metric):
            self.best_metric = metric
            self.worse_times = 0
            for key, value in kwargs.items():
                self.best_info[key] = value
            return True
        else:
            self.worse_times += 1
            return False

    def reach_stop(self) -> bool:
        """
        Checks if the stopping condition is met based on patience.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        return self.patience is not None and self.worse_times >= self.patience

    def __getitem__(self, key: str):
        """
        Retrieves the value associated with a given key from the internal dictionary.

        Args:
            key (str): The key to retrieve.

        Returns:
            Any: The value associated with the key.
        """
        return self.best_info[key]

    def __setitem__(self, key: str, value):
        self.best_info[key] = value

    def __delitem__(self, key: str):
        del self.best_info[key]

    def __contains__(self, key: str) -> bool:
        return key in self.best_info


class AverageMeter(Mapping[str, Any]):
    """
    A utility class for tracking and calculating the average of multiple metrics over a sequence of iterations.
    The `AverageMeter` is designed for use during training loops to track metrics such as losses or accuracies.

    Example:

        >>> from fedbox.utils.training import AverageMeter
        >>> meter = AverageMeter()
        >>> for epoch in range(epochs):
        ...     meter.clear()  # Clear the meter at the beginning of each epoch
        ...     for batch in dataloader:
        ...         cls_loss = ...  # Classification loss
        ...         reg_loss = ...  # Regularization loss
        ...         loss = cls_loss + reg_loss  # Total loss
        ...         optimizer.zero_grad()
        ...         loss.backward()
        ...         optimizer.step()
        ...         # Add the loss values to the meter
        ...         meter.add(
        ...             loss=loss.item(),
        ...             cls_loss=cls_loss.item(),
        ...             reg_loss=reg_loss.item()
        ...         )
        ...     # Print the average loss values of the epoch
        ...     print(
        ...         f'epoch {epoch}' +
        ...         f', loss: {meter["loss"]:.4f}' +
        ...         f', cls_loss: {meter["cls_loss"]:.4f}' +
        ...         f', reg_loss: {meter["reg_loss"]:.4f}'
        ...     )
    """
    def __init__(self):
        """
        Initializes an empty `AverageMeter`.
        """
        self.__sum_count: dict[str, tuple[Any, int]] = {}

    def add(self, **values):
        """
        Adds new values to the meter. For each key in `values`, updates the sum and count.

        Args:
            **values: Keyword arguments where the key is the metric name, and the value is the metric value to add.

        Example:
            >>> cls_loss = ...  # classification loss
            >>> reg_loss = ...  # regularization loss
            >>> loss = cls_loss + reg_loss  # total loss
            >>> meter.add(
            ...     loss=loss.item(),
            ...     cls_loss=cls_loss.item(),
            ...     reg_loss=reg_loss.item()
            ... )
        """
        for key, value in values.items():
            entry = self.__sum_count.get(key)
            if entry is None:
                self.__sum_count[key] = (value, 1)
            else:
                tot, cnt = entry
                self.__sum_count[key] = (tot + value), (cnt + 1)

    def __getitem__(self, key: str):
        """
        Retrieves the average value of a specified metric.

        Args:
            key (str): The name of the metric.

        Returns:
            float: The average value of the specified metric.

        Raises:
            KeyError: If the metric is not being tracked.
        """
        tot, cnt = self.__sum_count[key]
        return tot / cnt

    def __delitem__(self, key: str):
        """
        Deletes a tracked metric.

        Args:
            key (str): The name of the metric to delete.
        """
        del self.__sum_count[key]

    def __contains__(self, key: str) -> bool:
        """
        Checks if a metric is being tracked.

        Args:
            key (str): The name of the metric.

        Returns:
            bool: True if the metric is being tracked, False otherwise.
        """
        return key in self.__sum_count

    def clear(self):
        """
        Resets all tracked metrics.

        Example:
            >>> meter = AverageMeter()
            >>> meter.add(loss=loss.item())
            >>> meter.clear()
            >>> len(meter)
            0
        """
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


def module_requires_grad_(
    module: Union[torch.nn.Module, Iterable[torch.Tensor]],
    mode: bool
):
    params = module.parameters() if isinstance(module, torch.nn.Module) else module
    for p in params:
        p.requires_grad_(mode)


def freeze_module(module: Union[torch.nn.Module, Iterable[torch.Tensor]]):
    module_requires_grad_(module, mode=False)


def unfreeze_module(module: Union[torch.nn.Module, Iterable[torch.Tensor]]):
    module_requires_grad_(module, mode=True)


EarlyStopper = Recorder
MeanDict = AverageMeter
