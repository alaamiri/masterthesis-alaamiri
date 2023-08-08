from .abs_searchspace import AbsSS

import xautodl  # import this lib -- "https://github.com/D-X-Y/AutoDL-Projects", you can use pip install xautodl
from xautodl.models import get_cell_based_tiny_net

class NasBench(AbsSS):
    def __init__(self, dataset):
        super(NasBench, self).__init__(["none",
                                         "skip_connect",
                                         "nor_conv_1x1",
                                         "nor_conv_3x3",
                                         "avg_pool_3x3"], dataset)

        self.NB_OPS = 6

    def get_model(self, operations):
        model = self.get_arch_id(operations)
        config = self.api.get_net_config(model, self.dataset)
        network = get_cell_based_tiny_net(config)

        return network

    def get_arch_id(self, operations):
        arch = self.arch_to_str(operations)
        model = self.api.query_index_by_arch(arch)
        return model

    def arch_to_str(self, operations):
        return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*operations)

    def get_nasbench_unique(self, operations):
        return self.arch_to_str(operations)
