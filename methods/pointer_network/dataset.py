import torch
from torch.utils.data import Dataset


class PackingDataset(Dataset):
    def __init__(self):
        self.data = self._sample_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tuples, bin_size = self.data[index]
        input_tuples = torch.tensor(input_tuples, dtype=torch.float32)
        return {"net_input": input_tuples, "bin_size": bin_size}

    def _sample_data(self):
        return [
            ([(1, 2), (2, 1), (2, 2)], (10, 10)),
            ([(3, 3), (4, 1), (1, 2), (3, 2), (1, 1)], (20, 20)),
            ([(4, 1), (1, 2), (2, 1)], (10, 10)),
            ([(3, 3), (4, 1), (1, 2), (3, 2)], (10, 10)),
        ]

    def _load_data(self):
        pass
