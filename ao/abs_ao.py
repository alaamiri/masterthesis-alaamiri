from abc import ABC, abstractmethod

class AbsAO(ABC):
    def __init__(self, ss) -> None:
        super().__init__()
        self.search_space = ss
    
    @abstractmethod
    def reset_param(self):
        pass
        
    @abstractmethod
    def generate_arch(self):
        pass
    
    @abstractmethod
    def update(self):
        pass