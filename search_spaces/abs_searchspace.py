from abc import ABC, abstractmethod


class AbsSS(ABC):
    def get_model(self, operations):
        pass

    def arch_to_str(self, operations):
        pass