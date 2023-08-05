import torch
import numpy as np

from copy import deepcopy
from typing import Optional

from evotorch.operators import CopyingOperator
from evotorch import Problem, SolutionBatch


class OrderBasedMutation(CopyingOperator):
    def __init__(self, problem: Problem, *,  mutation_probability: Optional[float] = None):
        super().__init__(problem)
        self._mutation_probability = 1.0 if mutation_probability is None else float(mutation_probability)

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        data = result.access_values()
        mutation_vector = torch.rand(len(data)) <= self._mutation_probability
        for i, mutation in enumerate(mutation_vector):
            if mutation:
                start_index = np.random.randint(0, len(data[i]))
                end_index = np.random.randint(start_index, len(data[i]))
                indices = torch.randperm(end_index - start_index + 1) + start_index
                data[i][start_index:end_index+1] = data[i][indices]
        data[:] = self._respect_bounds(data)
        return result
