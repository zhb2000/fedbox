import torch
import torch.nn

from ...utils.functional import assign
from ..localonly import LocalOnlyClient


class FedAvgClient(LocalOnlyClient):
    def fit(self, global_model: torch.nn.Module) -> torch.nn.Module:
        assign[self.model] = global_model
        return LocalOnlyClient.fit(self)
