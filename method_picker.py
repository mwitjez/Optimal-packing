import numpy as np
import torch

from evotorch.logging import WandbLogger, StdOutLogger
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import GaussianMutation, CutAndSplice

from visualization.visualization_2d import Plotter2d
from visualization.visualization_3d import Plotter3d
from methods.bottom_left_fill import BottomLeftPacker
from methods.deepest_bottom_left_fill import DeepestBottomLeftPacker
from methods.genetic_algorithm import CustomGeneticAlgorithm
from methods.evotorch_problem import PackingProblem
from methods.evotorch_pmx import PartiallyMappedCrossOver
from methods.evotorch_mpox import MultiParentOrderCrossOver
from data.data import Data
from utils.time_wrapper import timing


class MethodPicker:

    @timing
    @staticmethod
    def run_2d(problem_name="C1"):
        data = Data().data_2d[problem_name]
        packer = BottomLeftPacker(
            data["items"], data["bin_size"][0], data["bin_size"][1] + 10
        )
        chromosome_length = data["num_items"]
        population_size = 100
        parents_number = 10
        mutation_rate = 0.2
        num_generations = 50
        genetic_algorithm = CustomGeneticAlgorithm(
            parents_number, chromosome_length, mutation_rate, packer
        )
        best_chromosome = genetic_algorithm.run(num_generations, population_size)
        solution = packer.pack_rectangles(best_chromosome)
        plotter = Plotter2d(solution)
        plotter.plot()
        genetic_algorithm.plot_stats()

    @timing
    @staticmethod
    def run_evotorch_2d(problem_name="C1"):
        data = Data().data_2d[problem_name]
        packer = BottomLeftPacker(
            data["items"], data["bin_size"][0], data["bin_size"][1] + 10
        )
        problem = PackingProblem(data, packer)
        ga = GeneticAlgorithm(
            problem,
            popsize=100,
            operators=[
                MultiParentOrderCrossOver(parents_per_child=4, problem=problem, tournament_size=10),
            ],
        )
        WandbLogger(ga, project="optimal_packing")
        StdOutLogger(ga)
        ga.run(50)
        print("Solution with best fitness ever:", ga.status["best"])
        best_chromosome = np.array(ga.status["best"]).tolist()
        solution = packer.pack_rectangles(best_chromosome)
        plotter = Plotter2d(solution)
        plotter.plot()

    @timing
    @staticmethod
    def run_3d(problem_name="P8"):
        data = Data().data_3d[problem_name]
        packer = DeepestBottomLeftPacker(
            data["items"],
            data["bin_size"][0],
            data["bin_size"][1],
            data["bin_size"][2] + 10,
        )
        chromosome_length = data["num_items"]
        population_size = 100
        parents_number = 30
        mutation_rate = 0.4
        num_generations = 50
        genetic_algorithm = CustomGeneticAlgorithm(
            parents_number, chromosome_length, mutation_rate, packer
        )
        best_chromosome = genetic_algorithm.run(num_generations, population_size)
        solution = packer.pack_rectangles(best_chromosome)
        #solution.save_to_json()
        plotter = Plotter3d(solution)
        plotter.plot()
        genetic_algorithm.plot_stats()

    @timing
    @staticmethod
    def run_evotorch_3d(problem_name="P8"):
        data = Data().data_3d[problem_name]
        packer = DeepestBottomLeftPacker(
            data["items"],
            data["bin_size"][0],
            data["bin_size"][1],
            data["bin_size"][2] + 10,
        )
        problem = PackingProblem(data, packer)
        ga = GeneticAlgorithm(
            problem,
            popsize=100,
            operators=[
                MultiParentOrderCrossOver(parents_per_child=4, problem=problem, tournament_size=10),
            ],
        )
        WandbLogger(ga, project="optimal_packing")
        StdOutLogger(ga)
        ga.run(50)
        print("Solution with best fitness ever:", ga.status["best"])
        best_chromosome = np.array(ga.status["best"]).tolist()
        solution = packer.pack_rectangles(best_chromosome)
        plotter = Plotter3d(solution)
        plotter.plot()
