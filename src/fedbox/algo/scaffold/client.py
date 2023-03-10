import copy
from typing import Iterable, Optional, NamedTuple, Any

import torch
import torch.nn
import torch.optim
from tqdm import tqdm

from ...utils.functional import assign, model_aggregate
from ..commons import mixin
from ..commons.optim import NoScheduleLR
from .functional import make_control_variate
from .optim import ScaffoldOptimizer


class Response(NamedTuple):
    model: torch.nn.Module
    delta_control: torch.nn.Module


class ScaffoldClient(mixin.Evaluate, mixin.Client):
    def __init__(
        self,
        *,
        id: int,
        model: torch.nn.Module,
        train_loader: Iterable,
        train_sample_num: int,
        valid_loader: Optional[Iterable] = None,
        test_loader: Optional[Iterable] = None,
        # --- config ---
        local_epochs: int,
        lr: float,
        device: torch.device,
        **other_config
    ):
        self.id = id
        self.model = model  # local model
        # local control variate (named c_i in the paper)
        self.control = make_control_variate(model)
        for p in self.control.parameters():
            p.data.zero_()
        self.train_loader = train_loader
        self.train_sample_num = train_sample_num
        self.valid_loader = valid_loader  # local valid samples (for PFL)
        self.test_loader = test_loader  # local test samples (for PFL)
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = device
        for key, value in other_config.items():
            setattr(self, key, value)
        self.optimizer = self.configure_optimizer()
        self.scheduler = self.configure_scheduler()

    def configure_optimizer(self):
        return ScaffoldOptimizer(self.model.parameters(), lr=self.lr)

    def configure_scheduler(self):
        return NoScheduleLR()

    def fit(self, global_model: torch.nn.Module, global_control: torch.nn.Module) -> Response:
        assign[self.model] = global_model
        self.model.to(self.device)
        global_model.to(self.device)
        self.control.to(self.device)
        global_control.to(self.device)
        self.model.train()
        step_num = 0
        for epoch in tqdm(range(self.local_epochs), desc=f'client {self.id}', leave=False):
            for data, targets in tqdm(self.train_loader, desc=f'epoch {epoch}', leave=False):
                data, targets = data.to(self.device), targets.to(self.device)
                output = self.model(data)
                loss = torch.nn.functional.cross_entropy(output, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(global_control.parameters(), self.control.parameters())
                step_num += 1
            self.scheduler.step()
        # update local control variate
        old_control = copy.deepcopy(self.control)
        delta_control = copy.deepcopy(self.control)
        assign[self.control] = model_aggregate(
            lambda ci, c, yi, x: ci - c + (x - yi) / (step_num * self.lr),
            self.control, global_control, self.model, global_model
        )
        assign[delta_control] = model_aggregate(
            lambda new, old: new - old,
            self.control, old_control
        )
        self.model.cpu()
        global_model.cpu()
        self.control.cpu()
        global_control.cpu()
        delta_control.cpu()
        return Response(self.model, delta_control)

    def make_checkpoint(self) -> dict[str, Any]:
        return { **mixin.Client.make_checkpoint(self), 'control': self.control }

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        mixin.Client.load_checkpoint(self, checkpoint)
        self.control.load_state_dict(checkpoint['control'])
