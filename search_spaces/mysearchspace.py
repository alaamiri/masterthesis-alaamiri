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
from primitives import (ResNetBasicblock, ReLUConvBN, Identity, Zero, AvgPool1x1)
from cells import Cell
import numpy as np
from collections import OrderedDict


class MySeachSpace(nn.Module):
    def __init__(self, operations, N, in_channels, cell_channels, num_classes = 10):
        super(MySeachSpace, self).__init__()
        self.channels = cell_channels
        self.num_classes = num_classes
        self.model = self.set_model(in_channels, cell_channels, operations, N)
        print(self.model)

    def set_model(self, in_channels, cell_channels, operations, N):
        model = nn.Sequential(OrderedDict([
            ('pre_proc', self.pre_processing(in_channels, cell_channels[0])),
            ('cells_block_1', self.cells(cell_channels[0], operations, N)),
            ('res_block_1', self.residual_block(cell_channels[0], cell_channels[1])),
            ('cells_2', self.cells(cell_channels[1], operations, N)),
            ('res_block_2', self.residual_block(cell_channels[1], cell_channels[2])),
            ('cells_block_3', self.cells(cell_channels[2], operations, N)),
            ('post_proc', self.post_processing(cell_channels[2], self.num_classes))
        ]))

        return model

    def pre_processing(self, C_in, C_out):
        pp = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_out)
        )

        return pp

    def cells(self, C, operations, N):
        cells = [Cell(C, operations) for _ in range(N)]
        c = nn.Sequential(*cells)
        return c

    def residual_block(self, C_in, C_out, stride = 2):
        rb = ResNetBasicblock(C_in, C_out, stride, affine=True)
        return rb

    def post_processing(self, C_in, num_classes):
        pp = nn.Sequential(
            nn.BatchNorm2d(C_in),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C_in, num_classes),
        )

        return pp

    def forward(self,x):
        return self.model(x)

if __name__ == '__main__':
    operations = ['identity', 'conv_3x3', 'avgpool_1x1', 'conv_3x3']
    cell_channels = [16, 32, 64]
    model = MySeachSpace(operations,3,3,cell_channels)
    x = torch.rand(1,3,64,64)
    y = model(x)