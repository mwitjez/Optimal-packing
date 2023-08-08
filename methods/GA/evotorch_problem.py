import torch

from evotorch.tools import ObjectArray
from evotorch import Problem, SolutionBatch

from utils.base_packer import BasePacker


class PackingProblem(Problem):
    def __init__(self, items_number: int, packer: BasePacker):
        self.packer = packer
        super().__init__(
            objective_sense="max",
            solution_length=items_number,
            dtype=torch.int64
        )

    def _evaluate_batch(self, solutions: SolutionBatch):
        n = len(solutions)
        fitnesses = torch.empty(n, dtype=torch.float32)
        for i in range(n):
            sln_values = solutions[i].values
            max_height = self.packer.get_max_height(sln_values)
            packing_density = self.packer.get_packing_density(sln_values)
            if max_height is None or packing_density is None:
                fitness = 0
            else:
                fitness = 1000 / (max_height) ** 3 + packing_density
            fitnesses[i] = fitness

        solutions.set_evals(fitnesses)

    def _fill(self, values: ObjectArray):
        for i in range(len(values)):
            values[i] = torch.randperm(self.solution_length)
