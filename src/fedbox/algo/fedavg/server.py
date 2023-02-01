from typing import Iterable, Optional

import torch
import torch.nn
from tqdm import tqdm

from ...utils.functional import assign, model_average
from ..commons import mixin
from .client import FedAvgClient


class FedAvgServer(mixin.Evaluate, mixin.PersonalizedEvaluate, mixin.Server):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        valid_loader: Optional[Iterable] = None,
        test_loader: Optional[Iterable] = None,
        # --- config ---
        global_rounds: int,
        client_join_num: Optional[int] = None,
        device: torch.device,
        **other_config
    ):
        self.clients: list[FedAvgClient] = []
        self.model = model  # global model
        self.valid_loader = valid_loader  # global validation samples
        self.test_loader = test_loader  # global test samples
        self.current_round = 0
        self.global_rounds = global_rounds
        self.client_join_num = client_join_num
        self.device = device
        for key, value in other_config.items():
            setattr(self, key, value)

    def fit(self) -> torch.nn.Module:
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients: list[FedAvgClient] = self.sample_clients()
            client_weights = [c.train_sample_num for c in selected_clients]
            recv_models: list[torch.nn.Module] = []
            for client in tqdm(selected_clients, desc=f'round {self.current_round}', leave=False):
                recv = client.fit(global_model=self.model)
                recv_models.append(recv)
            assign[self.model] = model_average(recv_models, client_weights)
        return self.model
