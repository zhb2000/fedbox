from ..fedavg import FedAvgServer as FedAvgServer
from .client import FedProxClient


class FedProxServer(FedAvgServer):
    clients: list[FedProxClient]
