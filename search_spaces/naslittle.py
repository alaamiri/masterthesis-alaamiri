import torch
from torch import nn
from .primitives import ResNetBasicblock
from .cells import CellDAG
from collections import OrderedDict
from .abs_searchspace import AbsSS
from nats_bench import create

NATS_BENCH_TSS_PATH = "nats_bench_data/NATS-tss-v1_0-3ffb9-simple"

class NASLittle(AbsSS):
    def __init__(self, N, in_channels, cell_channels, dataset, num_classes = 10):
        super(NASLittle, self).__init__(["none",
                                         "skip_connect",
                                         "nor_conv_1x1",
                                         "nor_conv_3x3",
                                         "avg_pool_3x3"], dataset)
        self.N = N
        self.in_channels = in_channels
        self.channels = cell_channels
        self.num_classes = num_classes
        self.dataset = dataset
       
        #Operations per cell
        self.NB_OPS = 3

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
        cells = [CellDAG(C, operations) for _ in range(N)]
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

    def get_arch_id(self, operations):
        unique_str = self.get_nasbench_unique(operations)
        model = self.api.query_index_by_arch(unique_str)

        return model


    def arch_to_str(self, operations):
        return f'|{operations[0]}~0|+|{operations[1]}~1|+|{operations[2]}~2|'


    def get_nasbench_unique(self, operations):
        nas_bench_arch = f'|{operations[0]}~0|+|none~0|{operations[1]}~1|+|none~0|none~1|{operations[2]}~2|'
        #unique_str = self.api.get_unique_str(nas_bench_arch)

        return nas_bench_arch


if __name__ == '__main__':
    operations = ['skip_connect', 'nor_conv_3x3', 'avg_pool_3x3', 'nor_conv_3x3']
    cell_channels = [16, 32, 64]
    ss = NASMedium(3, 3, cell_channels)
    model = ss.get_model(operations)
    print(ss.arch_to_str(operations))
    unique_str = ss.get_nasbench_unique(operations)
    print(unique_str)
    id = ss.api.query_index_by_arch(unique_str)
    print(id)
    arch = ss.api.arch(id)
    print(arch)
    #ss.api.qu

    x = torch.rand(1, 3, 64, 64)
    y = model(x)
    print(y)
