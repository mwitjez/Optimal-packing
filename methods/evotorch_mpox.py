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
        parents_split = torch.split(parents, self.parents_per_child, dim=0)
        number_of_children = len(parents_split)
        children = torch.full((number_of_children, len(parents[0])), -1)
        crossover_points = torch.randint(1, len(parents[0]), (self.parents_per_child - 1,)).sort()[0]
        crossover_points = torch.cat([torch.tensor([0]), crossover_points, torch.tensor([len(parents[0])])])

        for n, parents in enumerate(parents_split):
            #step 1
            children[n][crossover_points[1]:crossover_points[2]] = parents[1][crossover_points[1]:crossover_points[2]]
            #step 2
            for j in range(2, self.parents_per_child):
                for x in range(crossover_points[j], crossover_points[j+1]):
                    if parents[j][x] not in children[n]:
                        children[n][x] = parents[j][x]
                    else:
                        for y in range(x, x+len(parents[j])):
                            current_index = y % len(parents[j])
                            if parents[j][current_index] not in children[n]:
                                children[n][x] = parents[j][current_index]
                                break
            #step 3
            for x in range(crossover_points[0], crossover_points[1]):
                if parents[0][x] not in children[n]:
                    children[n][x] = parents[0][x]
                else:
                    for y in range(x, x+len(parents[j])):
                        current_index = y % len(parents[j])
                        if parents[j][current_index] not in children[n]:
                            children[n][x] = parents[j][current_index]
                            break

        return self._make_children_batch(children)
