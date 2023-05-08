import os
import pickle
import math
import neat

import numpy as np
from itertools import chain
import utils.neat_visualize as visualize
from utils.bin import Bin


class NeatPacker:
    def __init__(self, data) -> None:
        self.data = data

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            items, bin_size = self.data["items"], self.data["bin_size"]
            bin = Bin(bin_size[0], bin_size[1])
            fitness = 0.0
            remaining_items = [[item.width, item.height] for item in items]
            for i in range(len(items)):
                inputs = [
                    *bin.map.flatten(),
                    *list(chain.from_iterable(remaining_items)),
                ]
                item, pos_x, pos_y = net.activate(inputs)
                item = self.choose_item(remaining_items, item)
                pos_x, pos_y = self.scale_position(bin, pos_x, pos_y)
                remaining_items[remaining_items.index(item)] = [0, 0]
                if self.is_valid_position(bin, item, pos_x, pos_y):
                    self.mark_positions(bin, item, pos_x, pos_y)
                    fitness += 5
                    fitness = self.calculate_final_fitness(bin, fitness)
                else:
                    break
            genome.fitness = fitness

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def choose_item(self, remaining_items, item):
        """Returns the item that is chosen by the network."""
        items  = list(filter(lambda x: x!=[0,0], remaining_items))
        item_indx = int(len(items) * self.sigmoid(item))
        return items[item_indx]

    def scale_position(self, bin, x, y):
        """Returns the scaled position of the rectangle."""
        scaled_x = int(bin.width * self.sigmoid(x))
        scaled_y = int(bin.height * self.sigmoid(y))
        return scaled_x, scaled_y

    def is_valid_position(self, bin, rectangle, x, y):
        """Returns True if the rectangle can be placed at the given position, False otherwise."""
        if bin.height <= y + rectangle[0] or bin.width <= x + rectangle[1]:
            return False
        for i in range(y, y + rectangle[0]):
            for j in range(x, x + rectangle[1]):
                if bin.map[i][j] == 1:
                    return False
        return True

    def mark_positions(self, bin, rectangle, x, y):
        """Marks the positions of the rectangle in the bin map."""
        bin.map[y : y + rectangle[0], x : x + rectangle[1]] = 1

    def calculate_final_fitness(self, bin, fitness):
        """Returns the final fitness of the genome."""
        max_height = bin.map.nonzero()[0].max() + 1
        total_area = bin.map.shape[0] * max_height
        ones_area = np.sum(bin.map)
        packing_density = ones_area / total_area
        fitness = fitness + 10 * packing_density + 10 / max_height
        return fitness

    def run(self):
        # Load the config file, which is assumed to live in
        # the same directory as this script.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, "config.txt")
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )

        pop = neat.Population(config)
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.StdOutReporter(True))

        winner = pop.run(self.eval_genomes, 100)

        # Save the winner.
        with open("winner", "wb") as f:
            pickle.dump(winner, f)

        """ visualize.plot_stats(stats, ylog=True, view=True, filename="fitness.svg")
        visualize.plot_species(stats, view=True, filename="speciation.svg")

        visualize.draw_net(config, winner, True)

        visualize.draw_net(
            config, winner, view=True, filename="winner-ctrnn.gv"
        )
        visualize.draw_net(
            config,
            winner,
            view=True,
            filename="winner-pruned.gv",
            prune_unused=True,
        )
        """