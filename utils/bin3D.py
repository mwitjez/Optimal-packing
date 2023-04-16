import numpy as np
from dataclasses import dataclass


@dataclass
class Bin3D:
    """
    A class representing the rectangular container that the items will be packed into.
    """
    width: int
    height: int
    depth: int
    items: list

    def __init__(self, width: int, height: int, depth: int):
        self.width = width
        self.height = height
        self.depth = depth
        self.items = []
        self.map = np.zeros((self.height, self.width, self.depth))
