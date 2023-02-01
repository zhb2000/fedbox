"""
Implementation of paper [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://proceedings.mlr.press/v119/karimireddy20a.html)
(ICML 2020).
"""

from .optim import ScaffoldOptimizer
from .server import ScaffoldServer
from .client import ScaffoldClient
