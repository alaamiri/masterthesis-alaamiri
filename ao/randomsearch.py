from .abs_ao import AbsAO
import random

class RandomSearch(AbsAO):
    def __init__(self, ss) -> None:
        super(RandomSearch, self).__init__(ss)

    def generate_arch(self, n_ops):
        arch_l = []
        for _ in range(n_ops):
            i = random.randint(0,len(self.search_space)-1)
            arch_l.append(self.search_space[i])

        return arch_l, 0
        
    def reset_param(self):
        return super().reset_param()
    
    def update(self, prob_l, reward):
        return None