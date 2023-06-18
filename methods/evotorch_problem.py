import torch

from evotorch.tools import ObjectArray
from evotorch import Problem, SolutionBatch
from methods.bottom_left_fill import BottomLeftPacker


class PackingProblem(Problem):
    def __init__(self, data):
        self.bottom_left_packer = BottomLeftPacker(data["items"], data["bin_size"][0], data["bin_size"][1]+10)
        super().__init__(
            objective_sense="max",
            solution_length=data["num_items"],
            dtype=torch.int32,
        )

    def _evaluate_batch(self, solutions: SolutionBatch):
        # Get the number of solutions
        n = len(solutions)

        # Allocate a PyTorch tensor that will store the fitnesses
        fitnesses = torch.empty(n, dtype=torch.float32)

        # Fitness is computed as the sum of numeric values stored
        # by a solution.
        for i in range(n):
            # Get the values stored by a solution (which, in the case of
            # this example, is a Python list, because we initialize them
            # so in the _fill method).
            sln_values = solutions[i].values
            max_height = self.bottom_left_packer.get_max_height(sln_values)
            packing_density = self.bottom_left_packer.get_packing_density(sln_values)
            if max_height is None or packing_density is None:
                fitness = 0
            else:
                fitness = 1000 / (max_height) ** 3 + packing_density
            fitnesses[i] = fitness

        # Set the fitnesses
        solutions.set_evals(fitnesses)

    def _fill(self, values: ObjectArray):
        # `values` is an empty tensor of shape (n, m) where n is the number
        # of solutions and m is the solution length.
        # The responsibility of this method is to fill this tensor.
        # In the case of this example, let us say that we wish the new
        # solutions to have values sampled from a standard normal distribution.
        for i in range(len(values)):
            values[i] = torch.randperm(self.solution_length)
