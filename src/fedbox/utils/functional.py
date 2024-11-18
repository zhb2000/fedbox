from typing import Union, Callable, Sequence, Mapping, cast, overload

import numpy
import torch
import torch.nn
from torch import Tensor
from torch.nn import Module


@overload
def model_named_zip(
    model1: Union[Module, Mapping[str, Tensor]],
    model2: Union[Module, Mapping[str, Tensor]],
    /
) -> dict[str, tuple[Tensor, Tensor]]: ...
@overload
def model_named_zip(
    model1: Union[Module, Mapping[str, Tensor]],
    model2: Union[Module, Mapping[str, Tensor]],
    model3: Union[Module, Mapping[str, Tensor]],
    /
) -> dict[str, tuple[Tensor, Tensor, Tensor]]: ...
@overload
def model_named_zip(
    model1: Union[Module, Mapping[str, Tensor]],
    model2: Union[Module, Mapping[str, Tensor]],
    model3: Union[Module, Mapping[str, Tensor]],
    model4: Union[Module, Mapping[str, Tensor]],
    /,
    *models: Union[Module, Mapping[str, Tensor]],
) -> dict[str, tuple[Tensor, ...]]: ...


def model_named_zip(*models: Union[Module, Mapping[str, Tensor]]) -> dict[str, tuple[Tensor, ...]]:
    mappings: list[Mapping[str, Tensor]] = [m.state_dict() if isinstance(m, Module) else m for m in models]
    keys = mappings[0].keys()
    return {name: tuple(m[name] for m in mappings) for name in keys}  # TODO convert to tensor?


@overload
def model_aggregate(
    aggregator: Callable[[Sequence[Tensor]], Tensor],
    models: Sequence[Union[Module, Mapping[str, Tensor]]],
    /
) -> dict[str, Tensor]: ...
@overload
def model_aggregate(
    aggregator: Callable[[Tensor, Tensor], Tensor],
    model1: Union[Module, Mapping[str, Tensor]],
    model2: Union[Module, Mapping[str, Tensor]],
    /
) -> dict[str, Tensor]: ...
@overload
def model_aggregate(
    aggregator: Callable[[Tensor, Tensor, Tensor], Tensor],
    model1: Union[Module, Mapping[str, Tensor]],
    model2: Union[Module, Mapping[str, Tensor]],
    model3: Union[Module, Mapping[str, Tensor]],
    /
) -> dict[str, Tensor]: ...
@overload
def model_aggregate(
    aggregator: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    model1: Union[Module, Mapping[str, Tensor]],
    model2: Union[Module, Mapping[str, Tensor]],
    model3: Union[Module, Mapping[str, Tensor]],
    model4: Union[Module, Mapping[str, Tensor]],
    /
) -> dict[str, Tensor]: ...
@overload
def model_aggregate(
    aggregator: Callable[..., Tensor],
    model1: Union[Module, Mapping[str, Tensor]],
    model2: Union[Module, Mapping[str, Tensor]],
    model3: Union[Module, Mapping[str, Tensor]],
    model4: Union[Module, Mapping[str, Tensor]],
    model5: Union[Module, Mapping[str, Tensor]],
    /,
    *models: Module,
) -> dict[str, Tensor]: ...


@torch.no_grad()
def model_aggregate(aggregator: Callable[..., Tensor], *args) -> dict[str, Tensor]:
    """
    The sequence version: ::

        models: list[Module] = ...
        result: Module = ...
        result.load_state_dict(model_aggregation(average, models))

    The unpacked version: ::

        ma: Module = ...
        mb: Module = ...
        result: Module = ...
        result.load_state_dict(model_aggregation(lambda a, b: (a + b) / 2, ma, mb))
    """
    result: dict[str, Tensor] = {}
    if len(args) == 1:
        for name, params in model_named_zip(*args[0]).items():
            result[name] = aggregator(params)
    else:
        assert len(args) >= 2
        for name, params in model_named_zip(*args).items():
            result[name] = aggregator(*params)
    return result


def model_average(
    models: Sequence[Union[Module, Mapping[str, Tensor]]],
    weights: Sequence[float],
    normalize: bool = True,
) -> dict[str, Tensor]:
    return model_aggregate(
        lambda params: weighted_average(params, weights, normalize),
        models
    )


def weighted_average(
    tensors: Sequence[torch.Tensor],
    weights: Union[Sequence[float], torch.Tensor, numpy.ndarray, None] = None,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute the weighted average of a sequence of tensors.

    Args:
        tensors (Sequence[torch.Tensor]): A sequence of tensors to be averaged. 
            All tensors must have the same shape.
        weights (Union[Sequence[float], torch.Tensor, numpy.ndarray, None]): 
            Weights associated with each tensor. If None, the function computes a 
            simple average. The number of weights must match the number of tensors.
        normalize (bool, optional): If True (default), the weights will be normalized 
            so that their sum equals 1. If False, the weights will be used as provided.

    Returns:
        torch.Tensor: A tensor representing the weighted average. The shape of the 
            output tensor is the same as the shape of the individual tensors in the input.

    Example:
        >>> import torch
        >>> tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        >>> weights = [0.3, 0.7]
        >>> result = weighted_average(tensors, weights)
        >>> print(result)
        tensor([2.4, 3.4])

    Notes:
        - The function handles tensors with arbitrary dimensions.
        - Weights are automatically converted to a tensor and broadcasted to match 
          the dimensions of the input tensors.
    """
    stacked = torch.stack(list(tensors))  # shape: (num_tensors, dim1, ..., dim_n)
    if weights is None:  # Compute simple average if weights are not provided
        return stacked.mean(dim=0)
    w = torch.as_tensor(weights, dtype=torch.float, device=stacked.device)  # shape: (num_tensors,)
    if len(weights) != len(tensors):
        raise ValueError("The number of weights must match the number of tensors.")
    if w.sum() <= 0:
        raise ValueError("The sum of weights must be greater than zero.")
    if normalize:
        w = w / w.sum()  # Normalize weights if required
    # Expand weights to match the dimensions of the stacked tensors
    w = w.view(-1, *[1 for _ in range(stacked.dim() - 1)])  # w.shape: (num_tensors, 1, ..., 1)
    # Compute the weighted sum along the first dimension (tensors dimension)
    return (stacked * w).sum(dim=0)  # shape: (dim1, ..., dim_n)


from ._deprecated import model_assign, assign, Assign
