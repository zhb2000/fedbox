from typing import Optional, Callable, Iterable

import torch
import torch.optim

from ..commons.typing import OptimParams


class ScaffoldOptimizer(torch.optim.Optimizer):
    def __init__(self, params: OptimParams,  lr: float) -> None:
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        super().__init__(params, {'lr': lr})

    @torch.no_grad()
    def step(
        self,
        global_control: Iterable[torch.Tensor],
        local_control: Iterable[torch.Tensor],
        closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p, c, ci in zip(group['params'], global_control, local_control):
                p: torch.Tensor
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']
        return loss
