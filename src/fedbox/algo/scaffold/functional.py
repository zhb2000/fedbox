import copy

import torch
import torch.nn


def make_control_variate(model: torch.nn.Module) -> torch.nn.Module:
    control = copy.deepcopy(model)
    for p in control.parameters():
        p.data.zero_()
    return control
