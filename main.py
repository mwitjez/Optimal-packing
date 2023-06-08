from visualization.visualization_2d import Plotter2d
from visualization.visualization_3d import Plotter3d

from methods.bottom_left_fill import BottomLeftPacker
from methods.deepest_bottom_left_fill import DeepestBottomLeftPacker
from methods.genetic_algorithm import GeneticAlgorithm
from methods.neat_training import NeatPackerTrainer
from methods.neat_packing import NeatPacker
from data.data import Data


def run_2d():
    data = Data()
    packer = BottomLeftPacker(data.C1["items"], data.C1["bin_size"][0], data.C1["bin_size"][1]+10)
    population_size = 50
    parents_number = 5
    chromosome_length = data.C1["num_items"]
    mutation_rate = 0.8
    num_generations = 100
    genetic_algorithm = GeneticAlgorithm(parents_number, chromosome_length, mutation_rate, packer)
    best_chromosome = genetic_algorithm.run(num_generations, population_size)
    solution = packer.pack_rectangles(best_chromosome)
    plotter = Plotter2d(solution)
    plotter.plot()
    genetic_algorithm.plot_stats()

def run_3d():
    data = Data()
    packer = DeepestBottomLeftPacker(data.P27["items"], data.P27["bin_size"][0], data.P27["bin_size"][1], data.P27["bin_size"][2]+10)
    population_size = 100
    parents_number = 30
    chromosome_length = data.P27["num_items"]
    mutation_rate = 0.4
    num_generations = 50
    genetic_algorithm = GeneticAlgorithm(parents_number, chromosome_length, mutation_rate, packer)
    best_chromosome = genetic_algorithm.run(num_generations, population_size)
    solution = packer.pack_rectangles(best_chromosome)
    plotter = Plotter3d(solution)
    plotter.plot()

def train_neat_2d():
    data = Data()
    trainer = NeatPackerTrainer(data.C1)
    trainer.run()

def run_neat_2d():
    data = Data()
    packer = BottomLeftPacker(data.C1["items"], data.C1["bin_size"][0], data.C1["bin_size"][1]+10)
    neat = NeatPacker(data.C1)
    sequence = neat.get_packing_sequence()
    solution = packer.pack_rectangles(sequence)
    plotter = Plotter2d(solution)
    plotter.plot()

if __name__ == "__main__":
    run_3d()
