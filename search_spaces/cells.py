import torch
from torch import nn
from .primitives import (ResNetBasicblock, ReLUConvBN, Identity, Zero, AvgPool1x1)

class CellFCDAG(nn.Module):
    def __init__(self, C, operations):
        super(CellFCDAG, self).__init__()
        self.set_model(C, operations)
        self.operations = operations
        #print(self.model)

    """def set_model(self, C, operations):
        ops = [self.set_op(op, C) for op in operations]
        c = nn.Sequential(*ops)

        return c"""

    def set_model(self, C, operations):
        ops = [self.set_op(op, C) for op in operations]
        c = nn.Sequential(*ops)
        #'|op_0~0|+|op_1~0|op_2~1|'
        self.op_0 = self.set_op(operations[0], C)
        self.op_1 = self.set_op(operations[1], C)
        self.op_2 = self.set_op(operations[2], C)
        #self.node_3 = self.set_op(operations[3], C)

        return c

    def get_arch_str(self):
        return f'|{self.operations[0]}~0|+|{self.operations[1]}~0|{self.operations[2]}~1|'

    def set_op(self, op, C):
        OPERATIONS = {'identity': Identity(),
                      'zero': Zero(stride=1),
                      'conv_3x3': ReLUConvBN(C, C, kernel_size=3, affine=False, track_running_stats=False),
                      'conv_1x1': ReLUConvBN(C, C, kernel_size=1, affine=False, track_running_stats=False),
                      'avgpool_1x1': AvgPool1x1(kernel_size=3, stride=1, affine=False)
                      }
        return OPERATIONS[op]

    def forward(self, x):
        out_0 = self.op_0(x)
        out_1 = self.op_1(x)
        out_3 = self.op_2(out_1)

        return out_1+out_3

class CellDAG(nn.Module):
    def __init__(self, C, operations):
        super(CellFCDAG, self).__init__()
        self.set_model(C, operations)
        self.operations = operations
        #print(self.model)

    def set_model(self, C, operations):
        ops = [self.set_op(op, C) for op in operations]
        c = nn.Sequential(*ops)
        #'|op_0~0|+|op_1~0|op_2~1|'
        self.op_0 = self.set_op(operations[0], C)
        self.op_1 = self.set_op(operations[1], C)
        self.op_2 = self.set_op(operations[2], C)
        self.op_3 = self.set_op(operations[3], C)

        return c

    def get_arch_str(self):
        return f'|{self.operations[0]}~0|+|{self.operations[1]}~1|+|{self.operations[2]}~2|'

    def set_op(self, op, C):
        OPERATIONS = {'identity': Identity(),
                      'zero': Zero(stride=1),
                      'conv_3x3': ReLUConvBN(C, C, kernel_size=3, affine=False, track_running_stats=False),
                      'conv_1x1': ReLUConvBN(C, C, kernel_size=1, affine=False, track_running_stats=False),
                      'avgpool_1x1': AvgPool1x1(kernel_size=3, stride=1, affine=False)
                      }
        return OPERATIONS[op]

    def forward(self, x):
        out_0 = self.op_0(x)
        out_1 = self.op_1(out_0)
        out_2 = self.op_2(out_1)
        out_3 = self.op_3(out_2)

        return out_3

if __name__ == '__main__':
    operations = ['identity', 'conv_3x3', 'avgpool_1x1']
    cell_channels = [16, 32, 64]
    cell = CellFCDAG(cell_channels[0], operations)
    x = torch.rand([16,16,3,3])
    #print(x)
    y = cell(x)
    #print(y)
    print(cell)
    print(cell.get_arch_str())