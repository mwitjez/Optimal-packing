import numpy as np
import torch

from evotorch.logging import WandbLogger, StdOutLogger
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import GaussianMutation, CutAndSplice

from visualization.visualization_2d import Plotter2d
from visualization.visualization_3d import Plotter3d
from methods.GA.bottom_left_fill import BottomLeftPacker
from methods.GA.deepest_bottom_left_fill import DeepestBottomLeftPacker
from methods.GA.genetic_algorithm import CustomGeneticAlgorithm
from methods.GA.evotorch_problem import PackingProblem
from methods.GA.evotorch_pmx import PartiallyMappedCrossOver
from methods.GA.evotorch_mpox import MultiParentOrderCrossOver
from methods.GA.evotorch_custom_mutation import OrderBasedMutation
from data.data import Data
from utils.time_wrapper import timing
from utils.data_generator2d import DataGenerator
from utils.rectangle import Rectangle


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
        mutation_rate = 0.8
        num_generations = 200
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
        problem = PackingProblem(data["num_items"], packer)
        ga = GeneticAlgorithm(
            problem,
            popsize=128,
            operators=[
                MultiParentOrderCrossOver(parents_per_child=4, problem=problem, tournament_size=16),
                OrderBasedMutation(problem=problem, mutation_probability=0.05),
            ],
            elitist=False
        )
        WandbLogger(ga, project="optimal_packing")
        StdOutLogger(ga)
        ga.run(100)
        print("Solution with best fitness ever:", ga.status["best"])
        best_chromosome = np.array(ga.status["best"]).tolist()
        solution = packer.pack_rectangles(best_chromosome)
        plotter = Plotter2d(solution)
        plotter.plot()

    @staticmethod
    def test2d_data(problem_name="C1"):
        dg = DataGenerator(1, 10, 10)
        data = dg.generate()[0]
        rectangles = [Rectangle(rectangle[0], rectangle[1]) for rectangle in data[0]]
        packer = BottomLeftPacker(
            rectangles, data[2][0], data[2][1] + 10
        )
        best_chromosome = data[1]
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
        problem = PackingProblem(data["num_items"], packer)
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
