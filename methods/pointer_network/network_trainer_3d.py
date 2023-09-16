import torch
from torch.utils.data import DataLoader, random_split
from evotorch.neuroevolution import NEProblem
from evotorch.logging import StdOutLogger, WandbLogger
from evotorch.algorithms import PGPE
from tqdm import tqdm
from packers.deepest_bottom_left_fill import DeepestBottomLeftPacker

from .pointer_network import PointerNet
from .dataset_3d import PackingDataset3d
from utils.cuboid import Cuboid

class NetworkTrainer_3d:
    def __init__(self) -> None:
        self.dataset = PackingDataset3d()
        # self.train_dataloader, self.test_dataloader = self.train_test_split()
        self.network_config = {
            "embedding_dim": 32,
            "hidden_dim": 64,
            "lstm_layers": 2,
            "dropout": 0,
            "problem_dimension": 3
        }
        self.wandb_config = {
            **self.network_config,
            "batch_size": 128,
            "pop_size": 32,
            "num_epochs": 10,
            "dataset_size": len(self.dataset),
            "used_train_test_split": False,
            "only_generated_data": True,
            "center_learning_rate": 0.01,
            "stdev_learning_rate": 0.1,
            "radius_init": 4.5,
            "max_speed": 4.5 / 15.
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
        return score/len(self.dataset)

    def _evaluate_pointers(self, pointers, cuboids, bin_size):
        if not self._is_full_sequence(pointers, len(cuboids)):
            return 0
        cuboids = [[int(x) for x in cuboid] for cuboid in cuboids.tolist()]
        cuboids = [Cuboid(*cuboids[i]) for i in pointers]
        packer = DeepestBottomLeftPacker(cuboids, *bin_size)
        packing_density, max_height = packer.get_packing_density(pointers)
        if max_height is None or packing_density is None:
            return 0
        score = (bin_size[2]/max_height) + packing_density
        return score

    def _is_full_sequence(self, lst, n):
        return all(i in lst for i in range(n))

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
            radius_init=self.wandb_config["radius_init"],
            center_learning_rate=self.wandb_config["center_learning_rate"],
            stdev_learning_rate=self.wandb_config["stdev_learning_rate"],
            optimizer_config = {
                'max_speed': self.wandb_config["max_speed"],
                'momentum': 0.9,
            },
            distributed=False,
        )
        StdOutLogger(searcher)
        WandbLogger(searcher, project="3D Neuroevolution packing v3", config=self.wandb_config)
        searcher.run(self.wandb_config["num_epochs"])
        self.trained_network = problem.parameterize_net(searcher.status["best"].values)

    def save_network(self):
        torch.save(self.trained_network.state_dict(), "trained_network3D.pt")

    def load_network(self, file_name: str):
        self.trained_network = PointerNet(**self.network_config)
        self.trained_network.load_state_dict(torch.load(file_name))
        self.trained_network.eval()
        return self.trained_network
