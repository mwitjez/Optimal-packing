import torch
import json

from methods.pointer_network.network_trainer_3d import NetworkTrainer_3d
from packers.deepest_bottom_left_fill import DeepestBottomLeftPacker
from utils.cuboid import Cuboid


class ApiPacker():
    def __init__(self, cuboids):
        self.cuboids = self._convert_cuboids(cuboids)

    def _convert_cuboids(self, cuboids):
        return [(int(c.width), int(c.height), int(c.depth)) for c in cuboids]

    def _scale_cuboids_to_volume(self, cuboids, target_total_volume):
        total_volume = sum(w * h * d for w, h, d in cuboids)
        scaling_factor = (target_total_volume / total_volume) ** (1/3)
        scaled_cuboids = [(int(w * scaling_factor), int(h * scaling_factor), int(d * scaling_factor)) for w, h, d in cuboids]
        return scaled_cuboids

    def pack_cuboids(self):
        bin_size = (10, 10, 20)
        trainer = NetworkTrainer_3d()
        network = trainer.load_network("trained_network3D-ok.pt")
        network_input = torch.tensor(self.cuboids).float().unsqueeze(0)
        network_input = torch.nn.functional.normalize(network_input, dim=1)
        _, solution = network(network_input)
        cuboids_objects = [Cuboid(c[0], c[1], c[2]) for c in self.cuboids]
        packer = DeepestBottomLeftPacker(
                cuboids_objects, *bin_size
        )
        solution = packer.pack_rectangles(list(range(len(self.cuboids))))
        return self._convert_result(solution)

    def _convert_result(self, solution):
        data = []
        for item in solution.items:
            data.append({
                "x": item.x,
                "y": item.y,
                "z": item.z,
                "width": item.width,
                "height": item.height,
                "depth": item.depth
            })
        return data
