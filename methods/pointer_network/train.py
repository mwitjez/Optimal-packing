import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from evotorch.neuroevolution import NEProblem
from evotorch.logging import StdOutLogger
from ..GA.bottom_left_fill import BottomLeftPacker
from evotorch.algorithms import PGPE, SNES
from utils.rectangle import Rectangle
from copy import deepcopy


import numpy as np
import argparse
from tqdm import tqdm

from .pointer_network import PointerNet
from .dataset import PackingDataset

dataset = PackingDataset()

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

@torch.no_grad()
def eval_model(network: PointerNet):
    iterator = tqdm(dataloader, unit="Batch")
    score = 0
    for i_batch, sample_batched in enumerate(iterator):
        input = Variable(sample_batched["net_input"])
        bin_size = tuple(tensor.item() for tensor in sample_batched["bin_size"]) # zmienic żeby bottom left packer obsługiwał tensory
        packer_inputs = get_packer_inputs(input)
        packer = BottomLeftPacker(packer_inputs, bin_size[0], bin_size[1] + 10)
        network_out = network(input)
        max_height = packer.get_max_height(network_out[1].squeeze())
        packing_density = packer.get_packing_density(network_out[1].squeeze())
        if max_height is None or packing_density is None:
            score -= 100
        else:
            score += 1000 / (max_height) ** 3 + packing_density
    print(score)
    return score

def get_packer_inputs(inputs):
    packer_inputs = []
    for input in inputs.squeeze():
        packer_inputs.append(Rectangle(int(input[0].item()), int(input[1].item())))
    return packer_inputs

def neuroevolution_training():
    problem = NEProblem(
        # The objective sense -- we wish to maximize the sign_prediction_score
        objective_sense="max",
        # The network is a Linear layer mapping 3 inputs to 1 output
        network=PointerNet,
        network_args={
            "embedding_dim": 128,
            "hidden_dim": 512,
            "lstm_layers": 2,
            "dropout": 0,
        },
        # Networks will be evaluated according to sign_prediction_score
        network_eval_func=eval_model,
    )
    searcher = PGPE(
        problem,
        popsize=50,
        radius_init=2.25,
        center_learning_rate=0.2,
        stdev_learning_rate=0.1,
    )
    logger = StdOutLogger(searcher)
    searcher.run(50)


#shit_trainig(params, False, model, dataloader)
neuroevolution_training()
