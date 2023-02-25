from typing import Iterable, Optional, Any

import torch
import torch.nn
from tqdm import tqdm

from ...utils.functional import assign, model_average, model_aggregate
from ..commons import mixin
from .client import ScaffoldClient
from .functional import make_control_variate


class ScaffoldServer(mixin.Evaluate, mixin.PersonalizedEvaluate, mixin.Server):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        valid_loader: Optional[Iterable] = None,
        test_loader: Optional[Iterable] = None,
        # --- config ---
        global_rounds: int,
        client_join_num: Optional[int] = None,
        **other_config
    ):
        self.clients: list[ScaffoldClient] = []
        self.model = model  # global model
        self.control = make_control_variate(model)  # global control variate (named c in the paper)
        self.valid_loader = valid_loader  # global validation samples
        self.test_loader = test_loader  # global test samples
        self.current_round = 0
        self.global_rounds = global_rounds
        self.client_join_num = client_join_num
        for key, value in other_config.items():
            setattr(self, key, value)

    def fit(self) -> torch.nn.Module:
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients: list[ScaffoldClient] = self.sample_clients()
            client_weights = [c.train_sample_num for c in selected_clients]
            recvs = []
            for client in tqdm(selected_clients, desc=f'round {self.current_round}', leave=False):
                recv = client.fit(global_model=self.model, global_control=self.control)
                recvs.append(recv)
            # aggregate models
            assign[self.model] = model_average([recv.model for recv in recvs], client_weights)
            # aggregate control variates
            delta_control = model_average([recv.delta_control for recv in recvs], client_weights)
            ratio = len(selected_clients) / len(self.clients)  # |S| / N
            assign[self.control] = model_aggregate(
                lambda c, delta_c: c + ratio * delta_c,
                self.control, delta_control
            )  # c <- c + |S| / N * delta_c
        return self.model

    def make_checkpoint(self) -> dict[str, Any]:
        return { **mixin.Server.make_checkpoint(self), 'control': self.control }

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        mixin.Server.load_checkpoint(self, checkpoint)
        self.control.load_state_dict(checkpoint['control'])
