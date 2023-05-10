from visualization.visualization_2d import Plotter2d
from visualization.visualization_3d import Plotter3d

from methods.bottom_left_fill import BottomLeftPacker
from methods.deepest_bottom_left_fill import DeepestBottomLeftPacker
from methods.genetic_algorithm import GeneticAlgorithm
from methods.myneat import NeatPacker
from data.data import Data


def run_2d():
    data = Data()
    packer = BottomLeftPacker(data.C2["items"], data.C2["bin_size"][0], data.C2["bin_size"][1]+100)
    population_size = 50
    parents_number = 5
    chromosome_length = data.C2["num_items"]
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
    packer = DeepestBottomLeftPacker(data.P64["items"], data.P64["bin_size"][0], data.P64["bin_size"][1], data.P64["bin_size"][2]+10)
    population_size = 100
    parents_number = 30
    chromosome_length = data.P64["num_items"]
    mutation_rate = 0.4
    num_generations = 50
    genetic_algorithm = GeneticAlgorithm(parents_number, chromosome_length, mutation_rate, packer)
    best_chromosome = genetic_algorithm.run(num_generations, population_size)
    solution = packer.pack_rectangles(best_chromosome)
    plotter = Plotter3d(solution)
    plotter.plot()
    genetic_algorithm.plot_stats()

def run_neat_2d():
    data = Data()
    packer = NeatPacker(data.C1)
    packer.run()

if __name__ == "__main__":
    run_3d()
