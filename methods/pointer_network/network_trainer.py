import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from evotorch.neuroevolution import NEProblem
from evotorch.logging import StdOutLogger, WandbLogger
from ..GA.bottom_left_fill import BottomLeftPacker
from evotorch.algorithms import PGPE, SNES
from tqdm import tqdm
from rectpack import newPacker
import numpy as np

from utils.rectangle import Rectangle
from .pointer_network import PointerNet
from .dataset import PackingDataset

class NetworkTrainer:
    def __init__(self) -> None:
        self.dataset = PackingDataset()
        self.train_dataloader, self.test_dataloader = self.train_test_split()
        self.network_config = {
                "embedding_dim": 128,
                "hidden_dim": 512,
                "lstm_layers": 2,
                "dropout": 0,
            }

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
            rectangles = sample_batched[0]
            _, pointers = network(rectangles)
            rectangles = rectangles.squeeze(0) #TODO batch
            rectangles = [rectangles[i] for i in pointers]
            bin_size = sample_batched[1].squeeze(0) #TODO batch
            packer = newPacker(sort_algo=None)
            packer.add_bin(*bin_size)
            for rec in rectangles[0]: #TODO batch
                packer.add_rect(*map(int, rec))
            packer.pack()
            rec_map = np.zeros((bin_size[0], bin_size[1]))
            for rect in packer[0]: #TODO batch?
                rec_map[rect.corner_bot_l.x:rect.corner_top_r.x, rect.corner_bot_l.y:rect.corner_top_r.y] = 1
            max_height = rec_map.nonzero()[0].max() + 1
            total_area = rec_map.shape[0] * max_height
            ones_area = np.sum(rec_map)
            packing_density = ones_area / total_area
            if packing_density:
                score += packing_density
        return score

    def train(self):
        problem = NEProblem(
            objective_sense="max",
            network=PointerNet,
            network_args=self.network_config,
            network_eval_func=self.eval_model,
            num_actors="max",
        )
        searcher = PGPE(
            problem,
            popsize=10,
            radius_init=2.25,
            center_learning_rate=0.2,
            stdev_learning_rate=0.1,
            distributed=True,
        )
        StdOutLogger(searcher)
        #WandbLogger(searcher, project="Neuroevolution packing")
        searcher.run(10)
        self.trained_network = problem.parameterize_net(searcher.status['center'])

    def save_network(self):
        torch.save(self.trained_network.state_dict(), "trained_network.pt")

    def load_network(self):
        self.trained_network = PointerNet(**self.network_config)
        self.trained_network.load_state_dict(torch.load("trained_network.pt"))
        self.trained_network.eval()
        return self.trained_network
