# FedBox

![Python version](https://img.shields.io/badge/python-3.9_|_3.10-blue?logo=python&logoColor=white) ![license](https://img.shields.io/github/license/zhb2000/fedbox)

FedBox is a toolbox for federated learning research.

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

## Non-IID Data Splitting Setups

The module `fedbox.utils.data` provides several non-IID data splitting setups for federated learning research.

### Quantity-Based Label Skew

This non-IID setup is first introduced in [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html). In this setup, each client only contains samples from a few classes.

The quantity-based label skew is implemented in the function `split_by_label`. The parameter `class_per_client` is used to control the number of classes in each client.

The following code demonstrates how to split the MNIST dataset into 100 clients, and each client only contains samples from 2 classes. We **split and save the indices of samples** instead of the samples themselves for each client, which gives us more flexibility. The splitting results are saved in a JSON file.

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
    mnist_labels = mnist.targets.numpy()
    mnist_indices = np.arange(len(mnist))
    results: list[tuple[np.ndarray, np.ndarray]] = split_by_label(
        mnist_indices,
        mnist_labels,
        client_num,
        class_per_client=class_per_client
    )
    with open(splitting_file, 'w') as file:
        json.dump([indices.tolist() for indices, _ in results], file, indent=4)
```

The following code demonstrates how to read the dataset subset for each client. We **read the sample indices** for each client from the JSON file, then **use `Subset` to build the local dataset** for each client.

```python
from torch.utils.data import Subset

def read_clients_dataset(mnist: MNIST, splitting_file: str) -> list[Subset]:
    """Read the dataset subset for each client."""
    with open(splitting_file) as file:
        results: list[list[int]] = json.load(file)
    return [Subset(mnist, indices) for indices in results]
```

### Distribution-Based Label Skew

This setup is first introduced in [Bayesian Nonparametric Federated Learning of Neural Networks](https://proceedings.mlr.press/v97/yurochkin19a.html). In this setup, each client is allocated a certain proportion of samples from each class according to the Dirichlet distribution. Changing $\alpha$, the concentration parameter of Dirichlet distributions, can vary the degree of non-IID. That is to say, a smaller $\alpha$ results in a higher non-IID degree.

The distribution-based label skew is implemented in the function `split_dirichlet_label`. The parameter `alpha` is used to control the concentration parameter of the Dirichlet distribution.

```python
import json
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from fedbox.utils.data import split_dirichlet_label

if __name__ == '__main__':
    client_num = 100
    alpha = 0.1
    splitting_file = 'mnist-splitting.json'
    mnist = MNIST(
        train=True,
        transform=Compose([ToTensor(), Normalize([0.5], [0.5])]) 
    )
    mnist_labels = mnist.targets.numpy()
    mnist_indices = np.arange(len(mnist))
    results: list[tuple[np.ndarray, np.ndarray]] = split_dirichlet_label(
        mnist_indices,
        mnist_labels,
        client_num,
        alpha=alpha
    )
    with open(splitting_file, 'w') as file:
        json.dump([indices.tolist() for indices, _ in results], file, indent=4)
```

### Quantity Skew

This setup is first introduced in [Federated Learning on Non-IID Data Silos: An Experimental Study](https://ieeexplore.ieee.org/document/9835537). In this setup, the size of the local dataset varies across clients.  Dirichlet distribution to allocate different amounts of data samples into each client.

The quantity skew is implemented in the function `split_dirichlet_quantity`.

## Model Aggregation
### Model Averaging

Use `model_average` to perform model averaging. The function supports both simple model averaging and weighted model averaging.

```python
from fedbox.utils.functional import model_average

global_model: Module = ...
local_models: list[Module] = ...

# simple model averaging
global_model.load_state_dict(model_average(local_models))

# weighted model averaging
weights = [0.5, 0.3, 0.2]
global_model.load_state_dict(model_average(local_models, weights))
```

### Custom Aggregation Operations

Use `model_aggregate` to implement custom aggregate operations.

Aggregate two models:

```python
from fedbox.utils.functional import model_aggregate

global_model: Module = ...
local_model_a: Module = ...
local_model_b: Module = ...

def custom_aggregate(a: Tensor, b: Tensor) -> Tensor:
    return (a + b) / 2

result = model_aggregate(
    custom_aggregate,
    local_model_a, local_model_b
)
global_model.load_state_dict(result)
```

Aggregate a sequence of models:

```python
global_model: Module = ...
local_models: list[Module] = ...

def custom_aggregate(params: Sequence[Tensor]) -> Tensor:
    return sum(params) / len(params)

result = model_aggregate(custom_aggregate, local_models)
global_model.load_state_dict(result)
```

## Training Utilities
### Average Meter

`AverageMeter` is a tool for tracking the average value of multiple metrics during the training process. For example, you can use it to track the average loss value of each epoch.

```python
from fedbox.utils.training import AverageMeter

meter = AverageMeter()
for epoch in range(epochs):
    meter.clear()  # clear the meter at the beginning of each epoch
    for batch in dataloader:
        cls_loss = ...  # classification loss
        reg_loss = ...  # regularization loss
        loss = cls_loss + reg_loss  # total loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # add the loss values to the meter
        meter.add(
            loss=loss.item(),
            cls_loss=cls_loss.item(),
            reg_loss=reg_loss.item()
        )
    # print the average loss values of the epoch
    print(
        f'epoch {epoch}' +
        f', loss: {meter["loss"]:.4f}' +
        f', cls_loss: {meter["cls_loss"]:.4f}' +
        f', reg_loss: {meter["reg_loss"]:.4f}'
    )
```

### Recorder

Use `Recorder` to perform early stopping during training processes.

```python
from fedbox.utils.training import Recorder

stopper = Recorder(higher_better=True, patience=10)
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

### Freeze and Unfreeze Model

Use `freeze_model` and `unfreeze_model` to freeze and unfreeze the model's parameters. `module_requires_grad_` is a helper function that sets the `requires_grad` attribute of the model's parameters.

```python
from fedbox.utils.training import freeze_model, unfreeze_model, module_requires_grad_

model: Module = ...

# freeze the model
freeze_model(model)

# unfreeze the model
unfreeze_model(model)

# set the 'requires_grad' attribute of the model's parameters
module_requires_grad_(model, True)
# Equivalent to
for param in model.parameters():
    param.requires_grad_(True)
```
