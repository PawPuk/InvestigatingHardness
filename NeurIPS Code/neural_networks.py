from typing import Dict, Set

# This code is a modified version from https://github.com/marco-gherardi/stragglers
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def latent_representation(self, x: Tensor) -> Tensor:
        pass

    def radii(self, data_loader: DataLoader, sustainability_mask: Set[int]) -> Dict[int, Tensor]:
        """ Computes the radii of the representation of class manifolds.

        :param data_loader: DataLoader containing the data used for experiments
        :param sustainability_mask: used to increase sustainability of the code. When an inversion point was found for
        class i then there is no point in computing the radii of the manifold of class i anymore.
        :return: dictionary containing information on the radii of class manifolds for the given model; keys are the
        indices, and they map to Tensors containing a single float
        """
        radii = {i: torch.tensor(0) for i in range(10) if i not in sustainability_mask}
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            with torch.no_grad():
                for i in set(range(10)).difference(sustainability_mask):
                    class_data = data[target == i]
                    class_data = self.latent_representation(class_data)
                    # normalization (mapping onto the unit sphere)
                    class_data = torch.nn.functional.normalize(class_data, dim=1)
                    # computation of the metric quantities
                    class_data_mean = torch.mean(class_data, dim=0)
                    radii[i] = torch.sqrt(torch.sum(torch.square(class_data - class_data_mean)) / class_data.shape[0])
        return radii


class SimpleNN(MyNN):
    def __init__(self, in_size: int = 784, depth: int = 2, width_hid: int = 20, latent: int = 1, out_size: int = 10):
        """

        :param in_size: size of input
        :param depth: number of hidden layers
        :param width_hid: width of hidden layers
        :param latent: ordinal number of hidden layer where the observables are computed
        :param out_size: size of output - number of classes
        """
        super().__init__()
        if not 0 < latent <= depth + 1:
            raise ValueError(f"The 'latent' parameter must be in (0, depth+1]; "
                             f"0 < {latent} <= {depth + 1} does not hold.")
        self.latent = latent
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_size, width_hid, bias=True)])
        for _ in range(depth - 1):
            self.layers.append(torch.nn.Linear(width_hid, width_hid, bias=True))
        self.layers.append(torch.nn.Linear(width_hid, out_size, bias=True))

    def latent_representation(self, x: Tensor) -> Tensor:
        x = x.view(-1, self.layers[0].in_features)
        for layer_index in range(self.latent):
            x = self.layers[layer_index](x)
            if layer_index < len(self.layers) - 1:
                x = torch.tanh(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.latent_representation(x)
        for layer_index in range(self.latent, len(self.layers)):
            x = self.layers[layer_index](x)
            if layer_index < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(800, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = self.fc(x)
        return x


class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
