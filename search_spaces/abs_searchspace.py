from abc import ABC, abstractmethod
from nats_bench import create
NATS_BENCH_TSS_PATH = "nats_bench_data/NATS-tss-v1_0-3ffb9-simple"

class AbsSS(ABC):
    @abstractmethod
    def __init__(self, op, dataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.OPERATIONS = op
        self.api = create(NATS_BENCH_TSS_PATH,
                          'tss',
                          fast_mode=True,
                          verbose=False)

    @abstractmethod
    def get_model(self, operations):
        pass
    
    @abstractmethod
    def arch_to_str(self, operations):
        pass
    
    
    def get_op(self):
        return self.OPERATIONS

    def get_score_from_api(self, model, epochs):
        # (5374, 89.16)
        # print(self.api.find_best("cifar10", "ori-test"))
        # (13714, 84.82)
        # print(self.api.find_best("cifar10-valid", "ori-test"))
        # (13714, 84.89199999023438)
        # print(self.api.find_best("cifar10-valid", "x-valid"))
        if self.dataset == 'cifar10':
            info = self.api.get_more_info(model, self.dataset + '-valid', hp=str(epochs))  # !!!
        else:
            info = self.api.get_more_info(model, self.dataset, hp=str(epochs))
        # print("not valid", info)
        # m_info = self.api.get_net_config(model,self.dataset)
        # print("kiki",self.api.get_cost_info(model, 'cifar10-valid', '12'))
        # print("net config", m_info)
        # info = self.api.get_more_info(model, self.dataset + "-valid")
        # print("valid", info)
        r = info['valid-accuracy'] / 100

        return r

    def get_info(self, model, epochs):
        if self.dataset == 'cifar10':
            info = self.api.get_more_info(model, self.dataset + '-valid', hp=str(epochs))  # !!!
            cost_info = self.api.get_cost_info(model, self.dataset + '-valid', hp=str(epochs))
        else:
            info = self.api.get_more_info(model, self.dataset, hp=str(epochs))
            cost_info = self.api.get_cost_info(model, self.dataset, hp=str(epochs))

        return info, cost_info