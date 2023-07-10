from .abs_ao import AbsAO
import random

class RandomSearch(AbsAO):
    def __init__(self, ss) -> None:
        super().__init__(ss)
    
    
    def generate_arch(self, n_ops):
        arch_l = []
        for _ in range(n_ops):
            i = random.randint(0,len(self.ss)-1)
            arch_l.append(self.ss[i])

        return arch_l, 0
        
    def reset_param(self):
        return super().reset_param()
    
    def update(self, prob_l, reward):
        return None