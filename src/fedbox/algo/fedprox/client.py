from typing import Iterable, Optional

import torch
import torch.nn
import torch.optim
from tqdm import tqdm

from ...utils.functional import assign
from ..commons import mixin
from ..commons.optim import NoScheduleLR
from .optim import FedProxOptimizer


class FedProxClient(mixin.Evaluate, mixin.Client):
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
        mu: float,
        device: torch.device,
        **other_config
    ):
        self.id = id
        self.model = model  # local model
        self.train_loader = train_loader
        self.train_sample_num = train_sample_num
        self.valid_loader = valid_loader  # local valid samples (for PFL)
        self.test_loader = test_loader  # local test samples (for PFL)
        self.local_epochs = local_epochs
        self.lr, self.mu = lr, mu
        self.device = device
        for key, value in other_config.items():
            setattr(self, key, value)
        self.optimizer = self.configure_optimizer()
        self.scheduler = self.configure_scheduler()

    def configure_optimizer(self):
        return FedProxOptimizer(self.model.parameters(), lr=self.lr, mu=self.mu)

    def configure_scheduler(self):
        return NoScheduleLR()

    def fit(self, global_model: torch.nn.Module) -> torch.nn.Module:
        assign[self.model] = global_model
        self.model.to(self.device)
        global_model.to(self.device)
        self.model.train()
        for epoch in tqdm(range(self.local_epochs), desc=f'client {self.id}', leave=False):
            for data, targets in tqdm(self.train_loader, desc=f'epoch {epoch}', leave=False):
                data, targets = data.to(self.device), targets.to(self.device)
                output = self.model(data)
                loss = torch.nn.functional.cross_entropy(output, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(global_model.parameters())
            self.scheduler.step()
        self.model.cpu()
        global_model.cpu()
        return self.model
