import numpy as np

from matplotlib import pyplot as plt

from methods.bottom_left_fill import BottomLeftPacker
from methods.deepest_bottom_left_fill import DeepestBottomLeftPacker
from methods.genetic_algorithm import GeneticAlgorithm
from data.data import Data


def run_2d(runs_number):
    data = Data()
    packer = BottomLeftPacker(data.C2["items"], data.C2["bin_size"][0], data.C2["bin_size"][1]+100)
    population_size = 50
    parents_number = 5
    chromosome_length = data.C2["num_items"]
    mutation_rate = 0.8
    num_generations = 100
    best_heights =  np.empty((0,num_generations), int)
    best_fintesses =  np.empty((0,num_generations), int)
    for i in range(runs_number):
        print("Run number: ", i)
        genetic_algorithm = GeneticAlgorithm(parents_number, chromosome_length, mutation_rate, packer)
        genetic_algorithm.run(num_generations, population_size)
        best_fintesses = np.append(best_fintesses, np.array([genetic_algorithm.best_fitness]), axis=0)
        best_heights = np.append(best_heights, np.array([genetic_algorithm.max_heights]), axis=0)
    best_heights = np.mean(best_heights, axis=0)
    best_fintesses = np.mean(best_fintesses, axis=0)
    plot_stats(best_fintesses, best_heights)


def run_3d(runs_number):
    data = Data()
    packer = DeepestBottomLeftPacker(data.P64["items"], data.P64["bin_size"][0], data.P64["bin_size"][1], data.P64["bin_size"][2]+10)
    population_size = 100
    parents_number = 30
    chromosome_length = data.P64["num_items"]
    mutation_rate = 0.4
    num_generations = 50
    best_heights =  np.empty((0,num_generations), int)
    best_fintesses =  np.empty((0,num_generations), int)
    for i in range(runs_number):
        print("Run number: ", i)
        genetic_algorithm = GeneticAlgorithm(parents_number, chromosome_length, mutation_rate, packer)
        genetic_algorithm.run(num_generations, population_size)
        best_fintesses = np.append(best_fintesses, np.array([genetic_algorithm.best_fitness]), axis=0)
        best_heights = np.append(best_heights, np.array([genetic_algorithm.max_heights]), axis=0)
    best_heights = np.mean(best_heights, axis=0)
    best_fintesses = np.mean(best_fintesses, axis=0)
    plot_stats(best_fintesses, best_heights)

def plot_stats(fitness, heights):
    """Plots the best fitness and max heightsvalues for each generation."""
    fig, axs = plt.subplots(2, 1,)
    axs[0].plot(fitness, label="Best fitness")
    axs[1].plot(heights, label="Max height")
    axs[0].legend()
    axs[1].legend()
    plt.xlabel("Generation")
    plt.show()


if __name__ == "__main__":
    run_3d(10)
