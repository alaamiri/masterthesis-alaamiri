import itertools

import numpy as np

from plot import *

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data as data_utils
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim

class Peephole(nn.Module):
    def __init__(self):
        super(Peephole, self).__init__()

        self.lstm = nn.LSTM(40, 160, num_layers=2)
        self.mlp = self._init_mlp()

    def _init_mlp(self):
        # batch norm & ReLU
        # in 160 out 1
        pass