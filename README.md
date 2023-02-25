# FedBox

![Python version](https://img.shields.io/badge/python-3.9_|_3.10-blue?logo=python&logoColor=white) ![license](https://img.shields.io/github/license/zhb2000/fedbox)

Toolbox for federated learning research.

This project is still under development.🚧

# Installation

Install from source:

```shell
git clone https://github.com/zhb2000/fedbox.git
cd fedbox
pip install .
```

Editable installation (for development):

```shell
pip install --editable .
```

# Usage
## Model Aggregation

It's easy to implement a custom aggregate operation using `model_aggregate` and `assign`.

```python
from fedbox.utils.functional import assign, model_aggregate

ma: Module = ...
mb: Module = ...
result: Module = ...
assign[result] = model_aggregate(lambda a, b: (a + b) / 2, ma, mb)
```

The sequence version of `model_aggregate`:

```python
def average(params: Sequence[Tensor]) -> Tensor:
    return sum(params) / len(params)

models: list[Module] = ...
result: Module = ...
assign[result] = model_aggregation(average, models)
```

## Data Splitting

The module `fedbox.utils.data` provides several data splitting strategies for federated learning research.

The following example implements the non-IID setting of the paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html). Each client only has samples of two labels.

```python
import json
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from fedbox.utils.data import split_by_label, DatasetSubset

def split_to_clients(mnist: MNIST):
    results = split_by_label(
        np.arange(len(mnist)),  # indices
        mnist.targets.numpy(),  # labels
        client_num=100,
        class_per_client=2
    )  # split indices into 100 clients
    with open('mnist.json', 'w') as file:
        json.dump([indices.tolist() for indices, _ in results], file, indent=4)

def read_client_datasets(mnist: MNIST) -> list[DatasetSubset]:
    with open('mnist.json') as file:
        results: list[list[int]] = json.load(file)
    return [DatasetSubset(mnist, indices) for indices in results]

if __name__ == '__main__':
    mnist = MNIST(
        train=train,
        transform=Compose([ToTensor(), Normalize([0.5], [0.5])]) 
    )
    split_to_clients(mnist)
```

## Training Utilities

Use `EarlyStopper` to perform early stopping during training processes.

```python
from fedbox.utils.training import EarlyStopper

stopper = EarlyStopper(higher_better=True, patience=10)
for epoch in range(epochs):
    train(...)
    f1, acc = validate(...)
    # use 'f1' as the early stopping metric, and record the corresponding 'acc' and 'epoch'
    is_best = stopper.update(f1, acc=acc, epoch=epoch)
    print(f'epoch {epoch}, is best: {is_best}, f1: {f1:.4f}, acc: {acc:.4f}, ')
    if stopper.reach_stop():
        break
# print final result
print(f'best epoch: {stopper["epoch"]}, best f1: {stopper.best_metric}, acc: {stopper["epoch"]}')
```

## Trainers

The module `fedbox.algo` provides trainers of some federated learning algorithms.

Notice: This module is still a work in progress.🚧
