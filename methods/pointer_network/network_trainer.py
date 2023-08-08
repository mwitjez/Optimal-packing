import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from evotorch.neuroevolution import NEProblem
from evotorch.logging import StdOutLogger
from ..GA.bottom_left_fill import BottomLeftPacker
from evotorch.algorithms import PGPE, SNES
from evotorch.neuroevolution import SupervisedNE

from tqdm import tqdm

from utils.rectangle import Rectangle
from .pointer_network import PointerNet
from .dataset import PackingDataset


class NetworkTrainer:
    def __init__(self) -> None:
        self.dataset = PackingDataset()
        self.train_dataset, self.test_dataset = self.train_test_split()

    def train_test_split(self, train_proportion: float = 0.8):
        train_size = int(train_proportion * len(self.dataset))
        test_size = len(self.dataset) - train_size

        train_dataset, test_dataset = random_split(
            self.dataset, [train_size, test_size]
        )

        return train_dataset, test_dataset

    def train(self):
        packing_problem = SupervisedNE(
            dataset=self.train_dataset,
            network=PointerNet,
            network_args={
                "embedding_dim": 128,
                "hidden_dim": 512,
                "lstm_layers": 2,
                "dropout": 0,
            },
            minibatch_size=32,
            loss_func=torch.nn.CrossEntropyLoss(),
        )
        searcher = SNES(packing_problem, popsize=50, radius_init=2.25)
        logger = StdOutLogger(searcher)
        searcher.run(500)
