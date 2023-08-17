import numpy as np
import torch

from evotorch.logging import WandbLogger, StdOutLogger
from evotorch.algorithms import GeneticAlgorithm
from rectpack import newPacker
from rectpack.maxrects import MaxRects

from visualization.visualization_2d import Plotter2d, NetwrokDataPlotter2d
from visualization.visualization_3d import Plotter3d
from methods.GA.bottom_left_fill import BottomLeftPacker
from methods.GA.deepest_bottom_left_fill import DeepestBottomLeftPacker
from methods.GA.genetic_algorithm import CustomGeneticAlgorithm
from methods.GA.evotorch_problem import PackingProblem
from methods.GA.evotorch_pmx import PartiallyMappedCrossOver
from methods.GA.evotorch_mpox import MultiParentOrderCrossOver
from methods.GA.evotorch_custom_mutation import OrderBasedMutation
from methods.pointer_network.network_trainer_2d import NetworkTrainer_2d
from methods.pointer_network.network_trainer_3d import NetworkTrainer_3d
from data.data import Data
from utils.time_wrapper import timing
from utils.data_generator2d import DataGenerator
from utils.data_generator3d import DataGenerator
from utils.rectangle import Rectangle
from utils.cuboid import Cuboid


class MethodPicker:

    @timing
    @staticmethod
    def run_2d(problem_name="C1"):
        data = Data().data_2d_ga[problem_name]
        packer = BottomLeftPacker(
            data["items"], data["bin_size"][0], data["bin_size"][1] + 10
        )
        chromosome_length = data["num_items"]
        population_size = 128
        parents_number = 10
        mutation_rate = 0.8
        num_generations = 20
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
        data = Data().data_2d_ga[problem_name]
        packer = BottomLeftPacker(
            data["items"], data["bin_size"][0], data["bin_size"][1] + 10
        )
        problem = PackingProblem(data["num_items"], packer)
        ga = GeneticAlgorithm(
            problem,
            popsize=200,
            operators=[
                MultiParentOrderCrossOver(parents_per_child=4, problem=problem, tournament_size=16),
                OrderBasedMutation(problem=problem, mutation_probability=0.8),
            ],
            elitist=False
        )
        WandbLogger(ga, project="optimal_packing")
        StdOutLogger(ga)
        ga.run(20)
        print("Solution with best fitness ever:", ga.status["best"])
        best_chromosome = np.array(ga.status["best"]).tolist()
        solution = packer.pack_rectangles(best_chromosome)
        plotter = Plotter2d(solution)
        plotter.plot()

    @staticmethod
    def test2d_data():
        dg = DataGenerator(1, 20, 20)
        data = dg.generate()[0]
        rectangles = [Rectangle(rectangle[0], rectangle[1]) for rectangle in data[0]]
        packer = BottomLeftPacker(
            rectangles, data[2][0], data[2][1] + 10
        )
        best_chromosome = data[1]
        solution = packer.pack_rectangles(best_chromosome)
        plotter = Plotter2d(solution)
        plotter.plot()

    @staticmethod
    def test3d_data():
        dg = DataGenerator(100, 10, 10, 10)
        data = dg.generate()
        for d in data:
            print(len(d[0]))
        data = data[0]
        rectangles = [Cuboid(c[0], c[1], c[2]) for c in data[0]]
        print(len(rectangles))
        packer = DeepestBottomLeftPacker(
            rectangles, data[1][0], data[1][1], data[1][2] + 10
        )
        solution = packer.pack_rectangles(list(range(len(rectangles))))
        plotter = Plotter3d(solution)
        plotter.plot()

    @timing
    @staticmethod
    def run_3d(problem_name="P8"):
        data = Data().data_3d_ga[problem_name]
        packer = DeepestBottomLeftPacker(
            data["items"],
            data["bin_size"][0],
            data["bin_size"][1],
            data["bin_size"][2],
        )
        chromosome_length = data["num_items"]
        population_size = 100
        parents_number = 30
        mutation_rate = 0.4
        num_generations = 2
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
            data["bin_size"][2],
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

    @timing
    @staticmethod
    def train_pointer_network_2d():
        trainer = NetworkTrainer_2d()
        trainer.train()
        trainer.save_network()

    @timing
    @staticmethod
    def train_pointer_network_3d():
        trainer = NetworkTrainer_3d()
        trainer.train()
        trainer.save_network()

    @staticmethod
    def run_pointer_network_2d(problem_name="C1"):
        data = Data().data_2d_network[problem_name]
        trainer = NetworkTrainer_2d()
        network = trainer.load_network("trained_network_expert-oath-9.pt")
        network_input = torch.tensor(data["items"]).float().unsqueeze(0)
        network_input = torch.nn.functional.normalize(network_input, dim=1)
        _, solution = network(network_input)
        print(solution)
        packer = newPacker(sort_algo=None, pack_algo=MaxRects)
        packer.add_bin(*data["bin_size"])
        rectangles = [data["items"][i] for i in solution.squeeze()]
        for rec in rectangles:
            packer.add_rect(*map(int, rec))
        packer.pack()
        plotter = NetwrokDataPlotter2d(packer[0], data["bin_size"])
        plotter.plot()

    @staticmethod
    def run_pointer_network_3d(problem_name="P8"):
        data = Data().data_3d_network[problem_name]
        trainer = NetworkTrainer_3d()
        network = trainer.load_network("trained_network3D.pt")
        network_input = torch.tensor(data["items"]).float().unsqueeze(0)
        network_input = torch.nn.functional.normalize(network_input, dim=1)
        _, solution = network(network_input)
        print(solution)
        cuboids = [Cuboid(c[0], c[1], c[2]) for c in data["items"]]
        packer = DeepestBottomLeftPacker(
            cuboids, *data["bin_size"]
        )
        solution = packer.pack_rectangles(solution.squeeze())
        plotter = Plotter3d(solution)
        plotter.plot()
