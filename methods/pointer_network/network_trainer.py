import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from evotorch.neuroevolution import NEProblem
from evotorch.logging import StdOutLogger
from ..GA.bottom_left_fill import BottomLeftPacker
from evotorch.algorithms import PGPE, SNES
from tqdm import tqdm

from utils.rectangle import Rectangle
from .pointer_network import PointerNet
from .dataset import PackingDataset


class NetworkTrainer:
    def __init__(self) -> None:
        self.dataset = PackingDataset()
        self.train_dataloader, self.test_dataloader = self.train_test_split()

    def train_test_split(self, train_proportion: float = 0.8):
        train_size = int(train_proportion * len(self.dataset))
        test_size = len(self.dataset) - train_size

        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])

        batch_size = 1
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    @torch.no_grad()
    def eval_model(self, network: PointerNet):
        iterator = tqdm(self.train_dataloader, unit="Batch")
        score = 0
        for i_batch, sample_batched in enumerate(iterator):
            input = Variable(sample_batched["net_input"])
            bin_size = tuple(
                tensor.item() for tensor in sample_batched["bin_size"]
            )  # zmienic żeby bottom left packer obsługiwał tensory
            packer_inputs = self._get_packer_inputs(input)
            packer = BottomLeftPacker(packer_inputs, bin_size[0], bin_size[1] + 10)
            network_out = network(input)
            max_height = packer.get_max_height(network_out[1].squeeze())
            packing_density = packer.get_packing_density(network_out[1].squeeze())
            if max_height is None or packing_density is None:
                score -= 100
            else:
                score += 1000 / (max_height) ** 3 + packing_density
        return score

    def _get_packer_inputs(self, inputs):
        packer_inputs = []
        for input in inputs.squeeze():
            packer_inputs.append(Rectangle(int(input[0].item()), int(input[1].item())))
        return packer_inputs

    def train(self):
        problem = NEProblem(
            objective_sense="max",
            network=PointerNet,
            network_args={
                "embedding_dim": 128,
                "hidden_dim": 512,
                "lstm_layers": 2,
                "dropout": 0,
            },
            network_eval_func=self.eval_model,
            num_actors="max",
        )
        searcher = PGPE(
            problem,
            popsize=50,
            radius_init=2.25,
            center_learning_rate=0.2,
            stdev_learning_rate=0.1,
            distributed=True,
        )
        logger = StdOutLogger(searcher)
        searcher.run(50)
