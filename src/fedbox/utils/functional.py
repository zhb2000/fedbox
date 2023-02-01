import sys
import warnings
from typing import Union, Iterable, Callable, Sequence, cast, overload

import torch
import torch.nn
from torch import Tensor

__all__ = [
    'model_assign',
    'assign',
    'model_zip',
    'model_aggregate',
    'model_average',
    'weighted_average'
]


def model_zip(*models: Union[torch.nn.Module, Iterable[Tensor]]) -> Iterable[tuple[Tensor, ...]]:
    iterables: list[Iterable[Tensor]] = [
        m.parameters() if isinstance(m, torch.nn.Module) else m for m in models
    ]
    kwargs = {'strict': True} if sys.version_info >= (3, 10) else {}  # TODO zip strict
    return zip(*iterables, **kwargs)


@torch.no_grad()
def model_assign(
    dest: Union[torch.nn.Module, Iterable[Tensor]],
    value: Union[torch.nn.Module, Iterable[Tensor]]
):
    if dest is value:
        warnings.warn('You are doing self assignment.')
    for p1, p2 in model_zip(dest, value):
        p1.data = p2.data


class AssignHelper:
    def __setitem__(
        self,
        dest: Union[torch.nn.Module, Iterable[Tensor]],
        value: Union[torch.nn.Module, Iterable[Tensor]]
    ):
        model_assign(dest, value)

    def __call__(
        self,
        dest: Union[torch.nn.Module, Iterable[Tensor]],
        value: Union[torch.nn.Module, Iterable[Tensor]]
    ):
        model_assign(dest, value)


assign = AssignHelper()
"""
A helper to assign model parameters. ::

    assign[dest] = value

is equivalent to ::

    model_assign(dest, value)
"""


@overload
def model_aggregate(
    aggregator: Callable[[tuple[Tensor, ...]], Tensor],
    models: Sequence[Union[torch.nn.Module, Iterable[Tensor]]],
    /
) -> Iterable[Tensor]: ...


@overload
def model_aggregate(
    aggregator: Callable[[Tensor, Tensor], Tensor],
    model1: Union[torch.nn.Module, Iterable[Tensor]],
    model2: Union[torch.nn.Module, Iterable[Tensor]],
    /
) -> Iterable[Tensor]: ...


@overload
def model_aggregate(
    aggregator: Callable[[Tensor, Tensor, Tensor], Tensor],
    model1: Union[torch.nn.Module, Iterable[Tensor]],
    model2: Union[torch.nn.Module, Iterable[Tensor]],
    model3: Union[torch.nn.Module, Iterable[Tensor]],
    /
) -> Iterable[Tensor]: ...


@overload
def model_aggregate(
    aggregator: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    model1: Union[torch.nn.Module, Iterable[Tensor]],
    model2: Union[torch.nn.Module, Iterable[Tensor]],
    model3: Union[torch.nn.Module, Iterable[Tensor]],
    model4: Union[torch.nn.Module, Iterable[Tensor]],
    /
) -> Iterable[Tensor]: ...


@overload
def model_aggregate(
    aggregator: Callable[..., Tensor],
    model1: Union[torch.nn.Module, Iterable[Tensor]],
    model2: Union[torch.nn.Module, Iterable[Tensor]],
    model3: Union[torch.nn.Module, Iterable[Tensor]],
    model4: Union[torch.nn.Module, Iterable[Tensor]],
    model5: Union[torch.nn.Module, Iterable[Tensor]],
    /,
    *models: Union[torch.nn.Module, Iterable[Tensor]],
) -> Iterable[Tensor]: ...


@torch.no_grad()
def model_aggregate(aggregator: Callable[..., Tensor], *args) -> Iterable[Tensor]:
    """
    Sequence version: ::

        # `s` is a tuple of tensor
        result = model_aggregate(lambda s: (s[0] + s[1]) / 2, [model_a, model_b])

    Unpacked version: ::

        # `a` and `b` are tensors
        result = model_aggregate(lambda a, b: (a + b) / 2, model_a, model_b)
    """
    if len(args) == 1:
        return (aggregator(params) for params in model_zip(*args[0]))
    else:
        assert len(args) >= 2
        return (aggregator(*params) for params in model_zip(*args))


def model_average(
    models: Sequence[Union[torch.nn.Module, Iterable[Tensor]]],
    weights: Sequence[float],
    normalize: bool = True,
) -> Iterable[Tensor]:
    return model_aggregate(
        lambda params: weighted_average(params, weights, normalize),
        models
    )


def weighted_average(
    tensors: Sequence[Tensor],
    weights: Sequence[float],
    normalize: bool = True
) -> Tensor:
    if normalize:
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]
    result = sum(x * w for x, w in zip(tensors, weights))
    return cast(Tensor, result)
