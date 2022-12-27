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

    def _init_net(self):
        layers = []
        for layer in self.string_layers:
            layers.append(nn.Conv2d(layer[]))

    def forward(self):
        pass