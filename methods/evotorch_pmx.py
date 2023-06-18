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

        # Allocate an empty SolutionBatch that will store the children
        childpop = SolutionBatch(self.problem, popsize=num_parents, empty=True)

        # Gain access to the decision values tensor of the newly allocated
        # childpop
        childpop_values = childpop.access_values()

        # Here we somehow fill `childpop_values` by recombining the parents.
        # The most common thing to do is to produce two children by
        # combining parents1[0] and parents2[0], to produce the next two
        # children parents1[1] and parents2[1], and so on.
        childpop_values[:] = ...

        # Return the child population
        return childpop