from visualization.visualization_2d import Plotter2d
from visualization.visualization_3d import Plotter3d

from methods.bottom_left_fill import BottomLeftPacker
from methods.deepest_bottom_left_fill import DeepestBottomLeftPacker
from methods.genetic_algorithm import GeneticAlgorithm
from data.data import Data


class MethodPicker:

    @staticmethod
    def run_2d(problem_name="C1"):
        data = Data().data_2d[problem_name]
        packer = BottomLeftPacker(data["items"], data["bin_size"][0], data["bin_size"][1]+10)
        chromosome_length = data["num_items"]
        population_size = 50
        parents_number = 5
        mutation_rate = 0.8
        num_generations = 100
        genetic_algorithm = GeneticAlgorithm(parents_number, chromosome_length, mutation_rate, packer)
        best_chromosome = genetic_algorithm.run(num_generations, population_size)
        solution = packer.pack_rectangles(best_chromosome)
        plotter = Plotter2d(solution)
        plotter.plot()
        genetic_algorithm.plot_stats()

    @staticmethod
    def run_3d(problem_name="P8"):
        data = Data().data_3d[problem_name]
        packer = DeepestBottomLeftPacker(data["items"], data["bin_size"][0], data["bin_size"][1], data["bin_size"][2]+10)
        chromosome_length = data["num_items"]
        population_size = 100
        parents_number = 30
        mutation_rate = 0.4
        num_generations = 50
        genetic_algorithm = GeneticAlgorithm(parents_number, chromosome_length, mutation_rate, packer)
        best_chromosome = genetic_algorithm.run(num_generations, population_size)
        solution = packer.pack_rectangles(best_chromosome)
        plotter = Plotter3d(solution)
        plotter.plot()
        genetic_algorithm.plot_stats()
