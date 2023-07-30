import torch

from typing import Tuple
from evotorch import SolutionBatch
from evotorch.operators import CrossOver


class PartiallyMappedCrossOver(CrossOver):
    def _do_cross_over(self, parents1: torch.Tensor, parents2: torch.Tensor) -> SolutionBatch:
        children1, children2 = torch.empty_like(parents1), torch.empty_like(parents2)
        for i, (parent1, parent2) in enumerate(zip(parents1, parents2)):
            point1, point2 = sorted(torch.randint(0, len(parent1), (2,)))
            child1, child2 = self._crossover(parent1, parent2, point1, point2)
            children1[i], children2[i] = child1, child2

        children = torch.cat([children1, children2], dim=0)
        return self._make_children_batch(children)

    def _crossover(
        self, parent1: torch.Tensor, parent2: torch.Tensor, point1: int, point2: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        child1, child2 = torch.full_like(parent1, -1), torch.full_like(parent2, -1)
        child1[point1:point2] = parent1[point1:point2]
        child2[point1:point2] = parent2[point1:point2]
        self._fill_missing_values(child1, parent2)
        self._fill_missing_values(child2, parent1)
        return child1, child2

    def _fill_missing_values(
        self, child: torch.Tensor, parent: torch.Tensor) -> None:
        filtered_values = parent[~torch.isin(parent, child)]
        child[child == -1] = filtered_values
