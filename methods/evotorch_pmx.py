import torch

from evotorch import SolutionBatch
from evotorch.operators import CrossOver


class PartiallyMappedCrossOver(CrossOver):
    def _do_cross_over(
        self,
        parents1: torch.Tensor,
        parents2: torch.Tensor,
    ) -> SolutionBatch:
        # parents1 is a tensor storing the decision values of the first
        # half of the chosen parents.
        # parents2 is a tensor storing the decision values of the second
        # half of the chosen parents.

        # We expect that the lengths of parents1 and parents2 are equal.
        assert len(parents1) == len(parents2)
        children1 = torch.empty_like(parents1)
        children2 = torch.empty_like(parents2)

        for i, (parent1, parent2) in enumerate(zip(parents1, parents2)):
            point1 = torch.randint(0, len(parent1), (1,))
            point2 = torch.randint(0, len(parent1), (1,))

            if point2 < point1:
                point1, point2 = point2, point1

            child1 = torch.full(parent1.shape, -1)
            child1[point1:point2] = parent1[point1:point2]
            filtered_values = parent2[~torch.isin(parent2, child1)]
            child1[child1 == -1] = filtered_values

            child2 = torch.full(parent2.shape, -1)
            child2[point1:point2] = parent2[point1:point2]
            filtered_values = parent1[~torch.isin(parent1, child2)]
            child2[child2 == -1] = filtered_values

            children1[i] = child1
            children2[i] = child2

        children = torch.cat([children1, children2], dim=0)

        # Write the children solutions into a new SolutionBatch, and return the new batch
        result = self._make_children_batch(children)
        return result
