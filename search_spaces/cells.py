import torch
from torch import nn
from primitives import (ResNetBasicblock, ReLUConvBN, Identity, Zero, AvgPool1x1)

class Cell(nn.Module):
    def __init__(self, C, operations):
        super(Cell, self).__init__()
        self.set_model(C, operations)
        #print(self.model)

    """def set_model(self, C, operations):
        ops = [self.set_op(op, C) for op in operations]
        c = nn.Sequential(*ops)

        return c"""

    def set_model(self, C, operations):
        ops = [self.set_op(op, C) for op in operations]
        c = nn.Sequential(*ops)

        self.node_0 = self.set_op(operations[0], C)
        self.node_1 = self.set_op(operations[1], C)
        self.node_2 = self.set_op(operations[2], C)
        self.node_3 = self.set_op(operations[3], C)

        return c

    def set_op(self, op, C):
        OPERATIONS = {'identity': Identity(),
                      'zero': Zero(stride=1),
                      'conv_3x3': ReLUConvBN(C, C, kernel_size=3, affine=False, track_running_stats=False),
                      'conv_1x1': ReLUConvBN(C, C, kernel_size=1, affine=False, track_running_stats=False),
                      'avgpool_1x1': AvgPool1x1(kernel_size=3, stride=1, affine=False)
                      }
        return OPERATIONS[op]

    """def forward(self, x):
        x = self.node_0(x)
        x = self.node_1(x)
        x = self.node_2(x)
        x = self.node_3(x)

        return x"""

    def forward(self, x):
        out_0 = self.node_0(x)
        out_1 = self.node_1(out_0)
        out_2 = self.node_2(out_0)
        out_3 = self.node_3(out_0+out_1+out_2)

        return x

if __name__ == '__main__':
    operations = ['identity', 'conv_3x3', 'avgpool_1x1', 'conv_3x3']
    cell_channels = [16, 32, 64]
    cell = Cell(cell_channels[0], operations)
    x = torch.rand([16,16,3,3])
    print(x)
    y = cell(x)
    print(y)