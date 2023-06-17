import torch
import math
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from evotorch import Problem
from evotorch.operators import OnePointCrossOver
from bottom_left_fill import BottomLeftPacker


class EvotorchGeneticAlgorithm:
    def __init__(self, data) -> None:
        self.data = data
        self.problem = Problem(
            "min",
            self._fitness_funnction,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            solution_length=self.data["num_items"],
        )

        self.ga = GeneticAlgorithm(
            self.problem,
            operators=[OnePointCrossOver],
            popsize=100,
        )
        self.logger = StdOutLogger(self.ga, interval=100)
        self.bottom_left_packer = BottomLeftPacker()

    def _fitness_function(self, chromosome):
        max_height = self.bottom_left_packer.get_max_height(chromosome)
        packing_density = self.bottom_left_packer.get_packing_density(chromosome)
        fitness = 1000/(max_height)**3 + packing_density
        if math.isnan(fitness) or math.isinf(fitness):
            fitness = 0
        return fitness

    def run(self):
        self.ga.run(5000)

