import torch
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from evotorch import Problem
from evotorch.operators import OnePointCrossOver
from methods.bottom_left_fill import BottomLeftPacker


class EvotorchGeneticAlgorithm:
    def __init__(self, data) -> None:
        self.data = data
        self.problem = Problem(
            "max",
            self._fitness_function,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            solution_length=self.data["num_items"],
            initial_bounds=(0, self.data["num_items"]-1),
            dtype=torch.int32,
        )
        self.ga = GeneticAlgorithm(
            self.problem,
            operators=[OnePointCrossOver(self.problem, tournament_size=4)],
            popsize=100,
        )
        self.logger = StdOutLogger(self.ga, interval=10)
        self.bottom_left_packer = BottomLeftPacker(data["items"], data["bin_size"][0], data["bin_size"][1]+10)

    def _fitness_function(self, chromosome: torch.Tensor):
        chromosome = chromosome.tolist()
        max_height = self.bottom_left_packer.get_max_height(chromosome)
        packing_density = self.bottom_left_packer.get_packing_density(chromosome)
        if max_height is None or packing_density is None:
            fitness = 0
        else:
            fitness = 1000/(max_height)**3 + packing_density
        return fitness

    def run(self):
        self.ga.run(50)
        print("Solution with best fitness ever:", self.ga.status["best"])
