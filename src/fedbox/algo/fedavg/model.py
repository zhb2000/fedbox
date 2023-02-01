import torch
import torch.nn


# https://github.com/katsura-jp/fedavg.pytorch/blob/43680267cf839fbf56eec599605bed46e00328e9/src/models/cnn.py
class CNN(torch.nn.Module):
    """A CNN with two 5x5 convolution layers. Input image resolution: 28 * 28."""

    def __init__(self, in_channels: int, class_num: int):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels=32,
            kernel_size=5,
            padding=0,
            stride=1,
            bias=True
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=0,
            stride=1,
            bias=True
        )
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, class_num)
        self.relu = torch.nn.ReLU(inplace=True)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP(torch.nn.Module):
    """A 2-hidden-layers MLP, referred to as "2NN" in the paper."""
    def __init__(self, in_features: int, class_num: int, hidden_dim: int = 200):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, class_num)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
