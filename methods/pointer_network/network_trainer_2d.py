import torch
from torch.utils.data import DataLoader, random_split
from evotorch.neuroevolution import NEProblem
from evotorch.logging import StdOutLogger, WandbLogger
from evotorch.algorithms import PGPE
from tqdm import tqdm
from rectpack import newPacker
from rectpack.maxrects import MaxRects
import numpy as np

from .pointer_network import PointerNet
from .dataset_2d import PackingDataset2d


class NetworkTrainer_2d:
    def __init__(self) -> None:
        self.dataset = PackingDataset2d()
        #self.train_dataloader, self.test_dataloader = self.train_test_split()
        self.network_config = {
            "embedding_dim": 32,
            "hidden_dim": 128,
            "lstm_layers": 2,
            "dropout": 0,
        }
        self.wandb_config = {
            **self.network_config,
            "batch_size": 32,
            "pop_size": 50,
            "num_epochs": 25,
            "dataset_size": len(self.dataset),
            "used_train_test_split": False,
            "added_generated_data": True
        }
        self.train_dataloader = DataLoader(
            self.dataset,
            batch_size=self.wandb_config["batch_size"],
            shuffle=True,
            collate_fn=self.custom_collate,
        )

    def train_test_split(self, train_proportion: float = 0.8):
        train_size = int(train_proportion * len(self.dataset))
        test_size = len(self.dataset) - train_size

        train_dataset, test_dataset = random_split(
            self.dataset, [train_size, test_size]
        )

        batch_size = self.wandb_config["batch_size"]
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.custom_collate,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.custom_collate,
        )
        return train_loader, test_loader

    def custom_collate(self, batch):
        # Group samples by their sequence length
        seq_lengths = [len(item[0]) for item in batch]
        unique_lengths = list(set(seq_lengths))
        batches = {length: [[], [], []] for length in unique_lengths}

        for seq, original_seq, bin_size in batch:
            seq_length = len(seq)
            batches[seq_length][0].append(seq)
            batches[seq_length][1].append(original_seq)
            batches[seq_length][2].append(bin_size)
        for seq_length in unique_lengths:
            batches[seq_length][0] = torch.stack(batches[seq_length][0])
        return batches

    @torch.no_grad()
    def eval_model(self, network: PointerNet):
        iterator = tqdm(self.train_dataloader, unit="Batch")
        score = 0
        for i_batch, batch in enumerate(iterator):
            for sampled_batched in batch.values():
                scaled_rectangles = sampled_batched[0]
                original_rectangles = sampled_batched[1]
                bin_sizes = sampled_batched[2]
                _, batch_pointers = network(scaled_rectangles)
                for i, pointers in enumerate(batch_pointers):
                    score += self._evaluate_pointers(
                        pointers, original_rectangles[i], bin_sizes[i]
                    )
        return score

    def _evaluate_pointers(self, pointers, rectangles, bin_size):
        rectangles = [rectangles[i] for i in pointers]
        packer = newPacker(sort_algo=None, pack_algo=MaxRects)
        packer.add_bin(*bin_size)
        for rec in rectangles:
            packer.add_rect(*map(int, rec))
        packer.pack()
        rec_map = np.zeros((bin_size[0], bin_size[1]))
        for rect in packer[0]:
            rec_map[
                rect.corner_bot_l.x : rect.corner_top_r.x,
                rect.corner_bot_l.y : rect.corner_top_r.y,
            ] = 1
        max_height = rec_map.nonzero()[0].max() + 1
        total_area = rec_map.shape[0] * max_height
        ones_area = np.sum(rec_map)
        packing_density = ones_area / total_area
        return packing_density

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
            popsize=self.wandb_config["pop_size"],
            radius_init=2.25,
            center_learning_rate=0.2,
            stdev_learning_rate=0.1,
            distributed=True,
        )
        StdOutLogger(searcher)
        #WandbLogger(searcher, project="Neuroevolution packing", config=self.wandb_config)
        searcher.run(self.wandb_config["num_epochs"])
        self.trained_network = problem.parameterize_net(searcher.status["center"])

    def save_network(self):
        torch.save(self.trained_network.state_dict(), "trained_network.pt")

    def load_network(self, file_name:str):
        self.trained_network = PointerNet(**self.network_config)
        self.trained_network.load_state_dict(torch.load(file_name))
        self.trained_network.eval()
        return self.trained_network
