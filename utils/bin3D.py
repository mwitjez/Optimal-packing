import json
import numpy as np


class Bin3D:
    """
    A class representing the rectangular container that the items will be packed into.
    """
    width: int
    height: int
    depth: int
    items: list

    def __init__(self, width: int, height: int, depth: int, items: list = []):
        self.width = width
        self.height = height
        self.depth = depth
        self.items = []
        self.map = np.zeros((self.height, self.width, self.depth))

    def save_to_json(self):
        """
        Saves the bin to a json file.
        """
        with open('bin.json', 'w') as f:
            json.dump(self.items, f, default=lambda o: o.__dict__, indent=4)
