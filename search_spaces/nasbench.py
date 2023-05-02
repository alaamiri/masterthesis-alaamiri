import torch
from torch import nn
from .primitives import ResNetBasicblock
from .cells import CellFCDAG
from collections import OrderedDict
from .abs_searchspace import AbsSS
from nats_bench import create


NATS_BENCH_TSS_PATH = "nats_bench_data/NATS-tss-v1_0-3ffb9-simple"

class NasBench(AbsSS):
    def __init__(self, api_path=NATS_BENCH_TSS_PATH):
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

    def get_model(self, operations):
        arch = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*operations)
        model = self.api.query_index_by_arch(arch)

        return model

    def get_score_from_api(self, model, dataset):
        info = self.api.get_more_info(model, dataset + "-valid")
        r = info['valid-accuracy'] / 100

        return r
