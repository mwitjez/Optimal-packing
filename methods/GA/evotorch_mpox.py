import torch

from evotorch import SolutionBatch
from evotorch.operators import CrossOver


class MultiParentOrderCrossOver(CrossOver):
    def __init__(self, parents_per_child: int, **kwargs):
        self.parents_per_child = parents_per_child
        super().__init__(**kwargs)

    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        parents = self._do_tournament(batch)
        return self._do_cross_over(parents)

    @torch.no_grad()
    def _do_tournament(self, batch: SolutionBatch) -> tuple:
        num_tournaments = len(batch)

        if self._problem.is_multi_objective and self._obj_index is None:
            ranks, _ = batch.compute_pareto_ranks(crowdsort=False)
            n_fronts = torch.amax(ranks) + 1
            ranks = (n_fronts - ranks).to(torch.float)
            ranks += self._problem.make_uniform(len(batch), dtype=self._problem.eval_dtype, device=batch.device) * 0.1
        else:
            ranks = batch.utility(self._obj_index, ranking_method="centered")
        indata = batch._data
        tournament_indices = self.problem.make_randint(
            (num_tournaments, self._tournament_size), n=len(batch), device=indata.device
        )
        tournament_ranks = ranks[tournament_indices]
        tournament_rows = torch.arange(0, num_tournaments, device=indata.device)
        parents = tournament_indices[tournament_rows, torch.argmax(tournament_ranks, dim=-1)]
        parents = indata[parents]
        return parents

    def _do_cross_over(self, parents: torch.Tensor) -> SolutionBatch:
        parent_groups = torch.split(parents, self.parents_per_child, dim=0)
        number_of_children = len(parent_groups)
        children = torch.full((number_of_children, len(parents[0])), -1)
        crossover_points = self._generate_crossover_points(parents)

        for i, parents in enumerate(parent_groups):
            children[i] = self._single_child_crossover(parents, crossover_points)

        return self._make_children_batch(children)

    def _generate_crossover_points(self, parents):
        crossover_points = torch.randint(1, len(parents[0]), (self.parents_per_child - 1,)).sort()[0]
        return torch.cat([torch.tensor([0]), crossover_points, torch.tensor([len(parents[0])])])

    def _single_child_crossover(self, parents: torch.Tensor, crossover_points: torch.Tensor) -> torch.Tensor:
        child = self._copy_segment_from_second_parent(parents, crossover_points)
        child = self._fill_remaining_positions(child, parents, crossover_points)
        child = self._fill_remaining_positions_from_first_parent(child, parents, crossover_points)
        return child

    def _copy_segment_from_second_parent(self, parents: torch.Tensor, crossover_points: torch.Tensor) -> torch.Tensor:
        child = torch.full((len(parents[0]),), -1)
        child[crossover_points[1]:crossover_points[2]] = parents[1][crossover_points[1]:crossover_points[2]]
        return child

    def _fill_remaining_positions(self, child: torch.Tensor, parents: torch.Tensor, crossover_points: torch.Tensor) -> torch.Tensor:
        for j in range(2, self.parents_per_child):
            child = self._fill_from_parent(child, parents, crossover_points, parent_index=j)
        return child

    def _fill_remaining_positions_from_first_parent(self, child: torch.Tensor, parents: torch.Tensor, crossover_points: torch.Tensor) -> torch.Tensor:
        return self._fill_from_parent(child, parents, crossover_points, parent_index=0)

    def _fill_from_parent(self, child: torch.Tensor, parents: torch.Tensor, crossover_points: torch.Tensor, parent_index: int) -> torch.Tensor:
        for x in range(crossover_points[parent_index], crossover_points[parent_index+1]):
            if parents[0][x] not in child:
                child[x] = parents[0][x]
            else:
                for y in range(x, x+len(parents[0])):
                    current_index = y % len(parents[0])
                    if parents[0][current_index] not in child:
                        child[x] = parents[0][current_index]
                        break
        return child
