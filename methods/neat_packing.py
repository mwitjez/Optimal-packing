import os
import pickle
import math
import neat

import numpy as np
from itertools import chain
import utils.neat_visualize as visualize
from .bottom_left_fill import BottomLeftPacker


class NeatPacker:
    def __init__(self, data) -> None:
        self.data = data
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, "config.txt")
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        with open("winner", "rb") as f:
            genome = pickle.load(f)
        self.net = neat.ctrnn.CTRNN.create(genome, config, 1)
        self.packer = BottomLeftPacker(
            self.data["items"], self.data["bin_size"][0], self.data["bin_size"][1] + 10
        )

    def get_packing_sequence(self):
        items, bin_size = self.data["items"], self.data["bin_size"]
        remaining_items = [[item.width, item.height] for item in items]
        sequence = []
        for _ in range(len(items)):
            inputs = list(chain.from_iterable(remaining_items))
            item = self.net.advance(inputs, 1, 1)
            item = self.choose_item(remaining_items, item[0])
            item_indx = remaining_items.index(item)
            remaining_items[item_indx] = [0, 0]
            sequence.append(item_indx)

        return sequence

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def choose_item(self, remaining_items, item):
        """Returns the item that is chosen by the network."""
        filtered_items = list(filter(lambda x: x != [0, 0], remaining_items))
        item_indx = int(len(filtered_items) * self.sigmoid(item))
        return filtered_items[item_indx]

    def scale_position(self, bin, x, y):
        """Returns the scaled position of the rectangle."""
        scaled_x = int(bin.width * self.sigmoid(x))
        scaled_y = int(bin.height * self.sigmoid(y))
        return scaled_x, scaled_y
