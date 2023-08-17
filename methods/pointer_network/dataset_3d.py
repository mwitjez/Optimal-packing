import torch
import glob
import json

from torch.utils.data import Dataset

from utils.data_generator3d import DataGenerator


class PackingDataset3d(Dataset):
    def __init__(self):
        self.data = self._generate_data()
        self._normalize_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        scaled_cuboids, original_cuboids, bin_size = self.data[index]
        scaled_cuboids = torch.tensor(scaled_cuboids, dtype=torch.float32)
        original_cuboids = torch.tensor(original_cuboids, dtype=torch.float32)
        bin_size = torch.tensor(bin_size, dtype=torch.int64)
        return scaled_cuboids, original_cuboids, bin_size

    def _normalize_data(self):
        for i, (items, bin_size) in enumerate(self.data):
            max_item_size = max(max(item) for item in items)
            self.data[i] = (
                [(item[0] / max_item_size, item[1] / max_item_size, item[2] / max_item_size) for item in items],
                (items),
                (bin_size[0], bin_size[1], bin_size[2]),
            )

    def _sample_data(self):
        return [
            ([(1, 2, 4), (2, 1, 3), (2, 2, 3)], (10, 10, 10)),
            ([(1, 2, 4), (2, 1, 1), (2, 2, 2)], (10, 10, 10)),
            ([(1, 2, 4), (2, 1, 1), (2, 2, 2)], (10, 10, 10)),
            ([(1, 2, 4), (2, 1, 1), (2, 2, 2), (1, 2, 4), (2, 1, 1), (2, 2, 2)], (10, 10, 10)),
            ([(1, 2, 4), (2, 1, 1), (2, 2, 2), (1, 2, 4), (2, 1, 1), (2, 2, 2)], (10, 10, 10)),
            ([(1, 2, 4), (2, 1, 1), (2, 2, 2), (1, 2, 4), (2, 1, 1), (2, 2, 2)], (10, 10, 10)),
        ]

    def _load_data_from_files(self):
        file_paths = glob.glob("data/3D_network_data/*")
        data = []
        for file_path in file_paths:
            with open(file_path) as file:
                file_data = json.load(file)
                bin_size = (
                        file_data["bin_size"][0],
                        file_data["bin_size"][1],
                        file_data["bin_size"][2],
                    )
                items = [
                        (item["depth"], item["width"], item["height"])
                        for item in file_data["items"]
                    ]
                data.append((items, bin_size))
        return data

    def _generate_data(self):
        d = DataGenerator(1000, 10, 10, 10)
        return d.generate()
