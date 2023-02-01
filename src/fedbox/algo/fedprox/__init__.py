"""
Implementation of paper [Federated Optimization in Heterogeneous Networks](https://proceedings.mlsys.org/paper/2020/hash/38af86134b65d0f10fe33d30dd76442e-Abstract.html)
(MLSys 2020).

Official implementation: https://github.com/litian96/FedProx
"""

from .optim import FedProxOptimizer
from .client import FedProxClient
from .server import FedProxServer
