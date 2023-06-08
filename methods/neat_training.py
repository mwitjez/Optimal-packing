import os
import pickle
import math
import neat

import numpy as np
from itertools import chain
import utils.neat_visualize as visualize
from .bottom_left_fill import BottomLeftPacker
from utils.bin import Bin


class NeatPackerTrainer:
    def __init__(self, data) -> None:
        self.data = data

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.ctrnn.CTRNN.create(genome, config, 1)
            items, bin_size = self.data["items"], self.data["bin_size"]
            self.packer = BottomLeftPacker(self.data["items"], self.data["bin_size"][0], self.data["bin_size"][1]+10)
            net.reset()
            fitness = 0.0
            remaining_items = [[item.width, item.height] for item in items]
            sequence = []
            for _ in range(len(items)):
                inputs = list(chain.from_iterable(remaining_items))
                item = net.advance(inputs, 1, 1)
                item = self.choose_item(remaining_items, item[0])
                item_indx = remaining_items.index(item)
                remaining_items[item_indx] = [0, 0]
                sequence.append(item_indx)

            fitness = self.calculate_final_fitness(sequence)
            genome.fitness = fitness

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

    def calculate_final_fitness(self, sequence):
        """Returns the final fitness of the genome."""
        max_height = self.packer.get_max_height(sequence)
        packing_density = self.packer.get_packing_density(sequence)
        fitness = packing_density
        if math.isnan(fitness) or math.isinf(fitness):
            fitness = 0
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

        winner = pop.run(self.eval_genomes, 10)

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
        