from typing import Dict, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class OldSimpleNN(nn.Module):
    def __init__(self, width, depth):
        super(OldSimpleNN, self).__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(28 * 28, width, bias=True)])
        for _ in range(depth - 1):
            self.layers.append(torch.nn.Linear(width, width, bias=True))
        self.layers.append(torch.nn.Linear(width, 10, bias=True))

    def forward(self, x):
        x = x.view(-1, self.layers[0].in_features)
        for layer_index in range(len(self.layers)):
            x = self.layers[layer_index](x)
            if layer_index < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


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
    def __init__(self, in_size: int, depth: int, width_hid: int, latent: int, out_size: int = 10):
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
