import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from evotorch.neuroevolution import NEProblem
from evotorch.logging import StdOutLogger
from ..GA.bottom_left_fill import BottomLeftPacker
from evotorch.algorithms import PGPE, SNES
from evotorch.neuroevolution import SupervisedNE
import numpy as np

from tqdm import tqdm

from utils.rectangle import Rectangle
from .pointer_network import PointerNet 
from .custom_loss import CustomLoss
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
        # TODO: Pamiętać o distrtibuted i num_actors
        packing_problem = SupervisedNE(
            dataset=self.train_dataset,
            network=PointerNet,
            network_args={
                "embedding_dim": 32,
                "hidden_dim": 64,
                "lstm_layers": 2,
                "dropout": 0.1,
            },
            minibatch_size=8,
            loss_func=CustomLoss()
        )
        searcher = SNES(packing_problem, popsize=100, radius_init=2.25, optimizer="adam")
        logger = StdOutLogger(searcher)
        searcher.run(100)
        self.trained_network = packing_problem.parameterize_net(searcher.status['center'])


    def test_network(self):
        accuracy = 0
        for data in self.test_dataset:
            input_tuples, solution = data
            input_tuples = input_tuples.unsqueeze(0)
            solution = solution.unsqueeze(0)
            _, pointers = self.trained_network(input_tuples)
            print(pointers.squeeze())
            if (solution.squeeze() == pointers.squeeze()).all():
                accuracy += 1
        accuracy /= len(self.test_dataset) * 100
        print(f"accuracy: {accuracy}")

    def train_network_alternative(self):
        model = PointerNet(128, 128, 2, 0.1)

        dataloader = DataLoader(self.dataset,
                        batch_size=2,
                        shuffle=True)
        CCE = torch.nn.CrossEntropyLoss()
        model_optim = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                lr=0.001)

        losses = []

        epoch_number = 10000
        for epoch in range(epoch_number):
            batch_loss = []
            iterator = tqdm(dataloader, unit='Batch')

            for i_batch, sample_batched in enumerate(iterator):
                iterator.set_description('Batch %i/%i' % (epoch+1, epoch_number))

                train_batch = Variable(sample_batched[0])
                target_batch = Variable(sample_batched[1])

                o, p = model(train_batch)
                o = o.contiguous().view(-1, o.size()[-1])

                target_batch = target_batch.view(-1)

                loss = CCE(o, target_batch)

                losses.append(loss.data)
                batch_loss.append(loss.data)

                model_optim.zero_grad()
                loss.backward()
                model_optim.step()

                iterator.set_postfix(loss='{}'.format(loss.data))

            iterator.set_postfix(loss=np.average(batch_loss))
