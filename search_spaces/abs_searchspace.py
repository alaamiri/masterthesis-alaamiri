from abc import ABC, abstractmethod


class AbsSS(ABC):
    @abstractmethod
    def __init__(self, op) -> None:
        super().__init__()
        self.OPERATIONS = op

    @abstractmethod
    def get_model(self, operations):
        pass
    
    @abstractmethod
    def arch_to_str(self, operations):
        pass
    
    
    def get_op(self):
        return self.OPERATIONS