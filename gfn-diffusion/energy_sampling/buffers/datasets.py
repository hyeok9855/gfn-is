import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, device: torch.device, dataset_size: int):
        super().__init__()
        self.data = torch.tensor([]).to(device)
        self.dataset_size = dataset_size

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]

    def add(self, new_batch: torch.Tensor):
        self.data = torch.cat([self.data, new_batch.detach()], dim=0)
        self.trim_if_needed()

    def trim_if_needed(self):
        if self.data.size(0) > self.dataset_size:
            self.data = self.data[self.data.size(0) - self.dataset_size :]  # FIFO

    def update(self, indices, new_batch: torch.Tensor):
        self.data[indices] = new_batch.detach()

    def reorder(self, indices):
        self.data = self.data[indices]


class ZipDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return [dataset[idx] for dataset in self.datasets]
