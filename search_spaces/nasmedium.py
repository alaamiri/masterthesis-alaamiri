import torch
from torch import nn
from .primitives import ResNetBasicblock
from .cells import CellFCDAG
from collections import OrderedDict
from .abs_searchspace import AbsSS


class NASMedium(AbsSS):
    def __init__(self, N, in_channels, cell_channels, num_classes = 10):
        super(NASMedium, self).__init__()
        self.N = N
        self.in_channels = in_channels
        self.channels = cell_channels
        self.num_classes = num_classes
        #self.model = self.set_model(in_channels, cell_channels, operations, N)

        self.OPERATIONS = ['identity',
                           'zero',
                           'conv_3x3',
                           'conv_1x1',
                           'avgpool_1x1']
        #Operations per cell
        self.nb_op = 3

    def get_model(self, operations):
        model = nn.Sequential(OrderedDict([
            ('pre_proc', self.pre_processing(self.in_channels, self.channels[0])),
            ('cells_block_1', self.cells(self.channels[0], operations, self.N)),
            ('res_block_1', self.residual_block(self.channels[0], self.channels[1])),
            ('cells_2', self.cells(self.channels[1], operations, self.N)),
            ('res_block_2', self.residual_block(self.channels[1], self.channels[2])),
            ('cells_block_3', self.cells(self.channels[2], operations, self.N)),
            ('post_proc', self.post_processing(self.channels[2], self.num_classes))
        ]))

        return model


    def pre_processing(self, C_in, C_out):
        pp = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_out)
        )

        return pp

    def cells(self, C, operations, N):
        cells = [CellFCDAG(C, operations) for _ in range(N)]
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
            #nn.Softmax(dim=1)
        )

        return pp

    """def forward(self,x):
        return self.model(x)"""

if __name__ == '__main__':
    operations = ['identity', 'conv_3x3', 'avgpool_1x1', 'conv_3x3']
    cell_channels = [16, 32, 64]
    ss = NASMedium(3, 3, cell_channels)
    model = ss.get_model(operations)

    x = torch.rand(1,3,64,64)
    y = model(x)
    print(y)