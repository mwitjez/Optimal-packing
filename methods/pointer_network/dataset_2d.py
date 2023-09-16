import torch
import glob

from torch.utils.data import Dataset

from utils.data_generator2d import DataGenerator


class PackingDataset2d(Dataset):
    def __init__(self):
        self.data = self._load_data_from_files()
        self.data += self._generate_data()
        self._normalize_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        scaled_rectangles, original_rectangles, bin_size = self.data[index]
        scaled_rectangles = torch.tensor(scaled_rectangles, dtype=torch.float32)
        original_rectangles = torch.tensor(original_rectangles, dtype=torch.float32)
        bin_size = torch.tensor(bin_size, dtype=torch.int64)
        return scaled_rectangles, original_rectangles, bin_size

    def _normalize_data(self):
        for i, (items, bin_size) in enumerate(self.data):
            max_item_size = max(max(item) for item in items)
            self.data[i] = (
                [(item[0] / max_item_size, item[1] / max_item_size) for item in items],
                (items),
                (bin_size[0], bin_size[1]),
            )

    def _sample_data(self):
        return [
            ([(1, 2), (2, 1), (2, 2)], (10, 10)),
            ([(1, 2), (2, 1), (2, 2)], (10, 10)),
            ([(1, 2), (2, 1), (2, 2)], (10, 10)),
            ([(1, 2), (2, 1), (2, 2)], (10, 10)),
            ([(1, 2), (2, 1), (2, 2)], (10, 10)),
            ([(1, 2), (2, 1), (2, 2), (1, 2), (2, 1), (2, 2)], (10, 10)),
            ([(1, 2), (2, 1), (2, 2), (1, 2), (2, 1), (2, 2)], (10, 10)),
        ]

    def _load_data_from_files(self):
        file_paths = glob.glob("data/2D_network_data/*/*")
        data = []
        for file_path in file_paths:
            with open(file_path) as file:
                file_data = file.readlines()
                bin_size = tuple(int(size) for size in file_data[1].split())
                items = [tuple(map(int, line.split())) for line in file_data[2:]]
                data.append((items, bin_size))
        return data

    def _generate_data(self):
        d = DataGenerator(200, 50, 50)
        return d.generate()
