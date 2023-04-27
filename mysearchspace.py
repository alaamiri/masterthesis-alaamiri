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
import numpy as np
from collections import OrderedDict


class MySeachSpace(nn.Module):
    def __init__(self, operations, N, in_channels, cell_channels, num_classes = 10):
        self.channels = cell_channels
        self.num_classes = num_classes
        self.model = set_model(in_channels, cell_channels, operations)
        print(self.model)

    def set_model(self, in_channels, cell_channels, operations):
        model = nn.Sequential(OrderedDict([
            ('pre_proc', self.pre_processing(in_channels, cell_channels[0])),
            ('cells_1', self.cells(cell_channels[0], operations)),
            ('res_block_1', self.residual_block(cell_channels[0], cell_channels[1])),
            ('cells_2', self.cells(cell_channels[1], operations)),
            ('res_block_2', self.residual_block(cell_channels[1], cell_channels[2])),
            ('cells_3', self.cells(cell_channels[2], operations)),
            ('post_proc', self.post_processing(cell_channels[2], self.num_classes))
        ]))

        return model

    def pre_processing(self, C_in, C_out):
        pp = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_out)
        )

        return pp

    def set_op(self, op, C):
        OPERATIONS = {'identity': Identity(),
                      'zero': Zero(stride=1),
                      'conv_3x3': ReLUConvBN(C, C, kernel_size=3, affine=False, track_running_stats=False),
                      'conv_1x1': ReLUConvBN(C, C, kernel_size=1, affine=False, track_running_stats=False),
                      'avgpool_1x1': AvgPool1x1(kernel_size=3, stride=1, affine=False)
                      }
        return OPERATIONS(op)

    def cells(self, C, operations):
        ops = [set_op(op, C) for op in operations]
        c = nn.Sequential(*ops)

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
