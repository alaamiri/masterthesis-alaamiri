import itertools
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch import autograd
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

EPOCHS = 5

class Net(nn.Module):
    def __init__(self, string_layers):
        super(Net, self).__init__()
        self.expected_input_size = (1, 1, 28, 28)
        self.string_layers = string_layers

        self.net = self._init_net()
        self.out = self._init_linear()

        self.loss_fn = nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3) #optim.AdamP or SGDP or SGDW or SWATS
        self.optimizer = torch.optim.SGD(self.parameters(), lr=5e-2, momentum=0.9,
                                          weight_decay=1e-4, nesterov=True)

    def __repr__(self):
        return f"{self.net.__repr__()}\n{self.out.__repr__()}"

    def _init_net(self):
        layers = []
        prev_channel = 1
        for layer in self.string_layers:
            layers.append(nn.Conv2d(in_channels=prev_channel,
                                    out_channels=layer[2],
                                    kernel_size=(layer[0], layer[1]), #kernel_size=(layer[0], layer[1])
                                    stride=layer[3],    #stride=layer[3]
                                    padding=0))
            prev_channel = layer[2]
            layers.append(nn.BatchNorm2d(layer[2]))
            layers.append(nn.ReLU(inplace=True))
        layers.pop(-1)
        #layers.append(nn.AdaptiveAvgPool2d((3, 1)))
        return nn.Sequential(*layers)

    def _init_linear(self, output_features = 10):
        conv_out = self._get_output_shape(self.net, self.expected_input_size)
        # self.out = nn.Linear(prev_channel * 12 * 12, 10)
        return nn.Linear(conv_out, out_features=output_features)

    """
    code from https://discuss.pytorch.org/t/linear-layer-input-neurons-number-calculation-after-conv2d/28659/3
    """
    def _get_output_shape(self,model, image_dim):
        return np.prod(model(torch.rand(*(image_dim))).data.shape)


    def forward(self, x):
        y = self.net(x)
        #print(y.size())
        y = y.view(y.size(0),-1) #flatten
        #print(y.size())
        out_y = self.out(y)

        return out_y





