import itertools
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, string_layers):
        super(Net, self).__init__()
        self.string_layers = string_layers

        self.net = self._init_net()

        self.optimizer = optim.Adam(self.parameters(), lr=6e-4)

    def _init_net(self):
        layers = []
        prev_channel = 3
        for layer in self.string_layers:
            layers.append(nn.Conv2d(in_channels=prev_channel,
                                    out_channels=layer[2],
                                    kernel_size=(layer[0], layer[1]),
                                    stride=layer[3]))
            prev_channel = layer[2]

            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self):
        pass