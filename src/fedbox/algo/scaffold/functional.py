from typing import Union

import torch
import torch.nn


def make_control_variate(model: torch.nn.Module) -> list[torch.Tensor]:
    return [torch.zeros_like(p, requires_grad=True) for p in model.parameters()]


def control_to_(control: list[torch.Tensor], device: Union[torch.device, str]):
    for i in range(len(control)):
        control[i] = control[i].to(device)
