"""
Implementation of paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html)
(AISTATS 2017).
"""

from .server import FedAvgServer
from .client import FedAvgClient
from .model import CNN, MLP
