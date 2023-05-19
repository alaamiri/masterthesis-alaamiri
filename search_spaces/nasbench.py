from .abs_searchspace import AbsSS
from nats_bench import create
import xautodl  # import this lib -- "https://github.com/D-X-Y/AutoDL-Projects", you can use pip install xautodl
from xautodl.models import get_cell_based_tiny_net


NATS_BENCH_TSS_PATH = "nats_bench_data/NATS-tss-v1_0-3ffb9-simple"

class NasBench(AbsSS):
    def __init__(self, dataset, api_path=NATS_BENCH_TSS_PATH):
        self.dataset = dataset
        self.api = create(api_path,
                          'tss',
                          fast_mode=True,
                          verbose=False)

        self.OPERATIONS = ["none",
                           "skip_connect",
                           "nor_conv_1x1",
                           "nor_conv_3x3",
                           "avg_pool_3x3"]
        self.NB_OPS = 6

    """def get_model(self, operations):
        

        return model"""

    def get_model(self, operations):
        model = self.get_arch_id(operations)
        config = self.api.get_net_config(model, self.dataset)
        network = get_cell_based_tiny_net(config)

        return network

    def get_arch_id(self, operations):
        arch = self.arch_to_str(operations)
        model = self.api.query_index_by_arch(arch)
        return model

    def get_score_from_api(self, model):
        info = self.api.get_more_info(model, self.dataset + "-valid")
        r = info['valid-accuracy'] / 100

        return r

    def arch_to_str(self, operations):
        return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*operations)
