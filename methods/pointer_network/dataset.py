import torch
import glob

from torch.utils.data import Dataset, TensorDataset

from utils.data_generator2d import DataGenerator


class PackingDataset(Dataset):
    def __init__(self):
        self.data = self._load_data_from_files()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tuples, solution = self.data[index]
        input_tuples = torch.tensor(input_tuples, dtype=torch.float32)
        solution = torch.tensor(solution, dtype=torch.int64)
        return input_tuples, solution

    def _sample_data(self):
        return [
            ([(1, 2), (2, 1), (2, 2)], (10, 10)),
            ([(3, 3), (4, 1), (1, 2), (3, 2), (1, 1)], (20, 20)),
            ([(4, 1), (1, 2), (2, 1)], (10, 10)),
            ([(3, 3), (4, 1), (1, 2), (3, 2)], (10, 10)),
            ([(1, 2), (2, 1), (2, 2)], (10, 10)),
            ([(3, 3), (4, 1), (1, 2), (3, 2), (1, 1)], (20, 20)),
            ([(4, 1), (1, 2), (2, 1)], (10, 10)),
            ([(3, 3), (4, 1), (1, 2), (3, 2)], (10, 10)),
            ([(1, 2), (2, 1), (2, 2)], (10, 10)),
            ([(3, 3), (4, 1), (1, 2), (3, 2), (1, 1)], (20, 20)),
            ([(4, 1), (1, 2), (2, 1)], (10, 10)),
            ([(3, 3), (4, 1), (1, 2), (3, 2)], (10, 10)),
            ([(1, 2), (2, 1), (2, 2)], (10, 10)),
            ([(3, 3), (4, 1), (1, 2), (3, 2), (1, 1)], (20, 20)),
            ([(4, 1), (1, 2), (2, 1)], (10, 10)),
            ([(3, 3), (4, 1), (1, 2), (3, 2)], (10, 10)),
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
        d = DataGenerator(10, 10, 10)
        return d.generate()
