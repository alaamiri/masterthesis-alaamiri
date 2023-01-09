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


        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)

    def __repr__(self):
        return self.net.__repr__()

    def _init_net(self):
        layers = []
        prev_channel = 1
        for layer in self.string_layers:
            layers.append(nn.Conv2d(in_channels=prev_channel,
                                    out_channels=layer[2],
                                    kernel_size=(5, 5), #kernel_size=(layer[0], layer[1])
                                    stride=1,    #stride=layer[3]
                                    padding=0))
            prev_channel = layer[2]
            layers.append(nn.ReLU())
        self.out = nn.Linear(prev_channel * 12 * 12, 10)
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        #print(y.size())
        y = y.view(y.size(0),-1)
        #print(y.size())
        out_y = self.out(y)

        return out_y
