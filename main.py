from visualization.visualization_2d import Plotter2d
from methods.bottom_left_fill import BottomLeftPacker
from methods.genetic_algorithm import GeneticAlgorithm
from data.data import Data


if __name__ == "__main__":
    data = Data()
    bottom_left_packer = BottomLeftPacker(data.C1["items"], data.C1["bin_size"][0], data.C1["bin_size"][1]+5)

    population_size = 50
    parents_number = 5
    chromosome_length = data.C1["num_items"]
    mutation_rate = 0.5
    num_generations = 100
    genetic_algorithm = GeneticAlgorithm(parents_number, chromosome_length, mutation_rate, bottom_left_packer)
    # Run the genetic algorithm
    best_chromosome = genetic_algorithm.run(population_size, num_generations)
    # Plot the best chromosome
    solution = bottom_left_packer.pack_rectangles(best_chromosome)
    plotter = Plotter2d(solution)
    plotter.plot()
    genetic_algorithm.plot_stats()
