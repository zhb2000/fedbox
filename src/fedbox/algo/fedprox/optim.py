from typing import Optional, Callable, Iterable

import torch
import torch.optim

from ..commons.typing import OptimParams


class FedProxOptimizer(torch.optim.Optimizer):
    def __init__(self, params: OptimParams,  lr: float, mu: float) -> None:
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        super().__init__(params, {'lr': lr, 'mu': mu})

    @torch.no_grad()
    def step(
        self,
        global_params: Iterable[torch.Tensor],
        closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                p: torch.Tensor
                if p.grad is None:
                    continue
                g = g.to(p.device)
                dp: torch.Tensor = p.grad.data + group['mu'] * (p.data - g.data)
                p.data = p.data - dp.data * group['lr']
        return loss
