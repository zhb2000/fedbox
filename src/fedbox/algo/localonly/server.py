from typing import Iterable, Optional, Any

import torch
import torch.nn
from tqdm import tqdm

from ..commons import mixin
from .client import LocalOnlyClient


class LocalOnlyServer(mixin.Evaluate, mixin.PersonalizedEvaluate, mixin.Server):
    def __init__(
        self,
        *,
        valid_loader: Optional[Iterable] = None,
        test_loader: Optional[Iterable] = None,
        # --- config ---
        global_rounds: int,
        client_join_num: Optional[int] = None,
        device: torch.device,
        **other_config
    ):
        self.clients: list[LocalOnlyClient] = []
        self.valid_loader = valid_loader  # global validation samples
        self.test_loader = test_loader  # global test samples
        self.current_round = 0
        self.global_rounds = global_rounds
        self.client_join_num = client_join_num
        self.device = device
        for key, value in other_config.items():
            setattr(self, key, value)

    def fit(self):
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients: list[LocalOnlyClient] = self.sample_clients()
            for client in tqdm(selected_clients, desc=f'round {self.current_round}', leave=False):
                client.fit()

    @torch.no_grad()
    def evaluate(self, loader: Iterable) -> dict[str, Any]:
        """Local model's performance on global test data."""
        clients_acc = [client.evaluate(loader)['acc'] for client in self.clients]
        average_acc = sum(clients_acc) / len(clients_acc)
        return {'acc': average_acc}
