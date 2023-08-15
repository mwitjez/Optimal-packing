import numpy as np


class Bin:
    """
    A class representing the rectangular container that the items will be packed into.
    """
    width: int
    height: int
    items: list

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.items = []
        self.map = np.zeros((self.height, self.width))
