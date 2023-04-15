from visualization.visualization_2d import Plotter2d
from methods.bottom_left_fill import BottomLeftPacker
from methods.genetic_algorithm import GeneticAlgorithm
from data.data import Data


if __name__ == "__main__":
    data = Data()
    bottom_left_packer = BottomLeftPacker(data.P1_rectangles, 20, 40)

    population_size = 100
    parents_number = 10
    chromosome_length = len(data.P1_rectangles)
    mutation_rate = 0.1
    num_generations = 25
    genetic_algorithm = GeneticAlgorithm(population_size, parents_number, chromosome_length, mutation_rate, bottom_left_packer)
    # Run the genetic algorithm
    best_chromosome = genetic_algorithm.run(num_generations)
    # Plot the best chromosome
    solution = bottom_left_packer.pack_rectangles(best_chromosome)
    plotter = Plotter2d(solution)
    plotter.plot()
    genetic_algorithm.plot_stats()
