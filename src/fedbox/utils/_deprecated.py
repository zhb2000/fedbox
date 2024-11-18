from typing import Union, Mapping

import torch.nn
from torch import Tensor
from torch.nn import Module


def model_assign(dest: Module, src: Union[Module, Mapping[str, Tensor]]):
    if isinstance(src, Module):
        src = src.state_dict()
    dest.load_state_dict(src, strict=False)


class Assign:
    def __setitem__(self, dest: Module, src: Union[Module, Mapping[str, Tensor]]):
        model_assign(dest, src)

    __call__ = __setitem__


assign = Assign()
"""
A helper object to assign model parameters. ::

    assign[dest] = src

is equivalent to ::

    model_assign(dest, src)
"""
