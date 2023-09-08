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
# For an editable installation, replace 'pip install .' to
pip install --editable .
```

# Usage

## Data Splitting

The module `fedbox.utils.data` provides several data splitting strategies for federated learning research.

The following example implements the non-IID setting of the paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html). Each client only has samples of two labels.

Split the dataset:

```python
import json
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from fedbox.utils.data import split_by_label

if __name__ == '__main__':
    client_num = 100
    class_per_client = 2
    splitting_file = 'mnist-splitting.json'
    mnist = MNIST(
        train=True,
        transform=Compose([ToTensor(), Normalize([0.5], [0.5])]) 
    )
    results: list[tuple[np.ndarray, np.ndarray]] = split_by_label(
        np.arange(len(mnist)),  # indices
        mnist.targets.numpy(),  # labels
        client_num,
        class_per_client=class_per_client
    )  # split indices into 100 clients
    with open(splitting_file, 'w') as file:
        json.dump([indices.tolist() for indices, _ in results], file, indent=4)
```

Read the dataset:

```python
from torch.utils.data import Subset

def read_clients_dataset(mnist: MNIST, splitting_file: str) -> list[Subset]:
    """Read the dataset subset for each client."""
    with open(splitting_file) as file:
        results: list[list[int]] = json.load(file)
    return [Subset(mnist, indices) for indices in results]
```

## Model Aggregation

Use `model_aggregate` to implement custom aggregate operations.

Aggregate two models:

```python
from fedbox.utils.functional import model_aggregate

ma: Module = ...
mb: Module = ...
result: Module = ...
result.load_state_dict(model_aggregate(lambda a, b: (a + b) / 2, ma, mb))
```

Aggregate a sequence of models:

```python
def average(params: Sequence[Tensor]) -> Tensor:
    return sum(params) / len(params)

models: list[Module] = ...
result: Module = ...
result.load_state_dict(model_aggregate(average, models))
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
print(f'best f1: {stopper.best_metric}, best epoch: {stopper["epoch"]}, acc: {stopper["acc"]}')
```
