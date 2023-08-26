from abc import ABC, abstractmethod


class BasePacker(ABC):

    @abstractmethod
    def get_max_height(self):
        pass

    @abstractmethod
    def get_packing_density(self):
        pass
    @abstractmethod

    def pack_rectangles(self):
        pass
