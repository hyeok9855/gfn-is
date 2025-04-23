from math import prod

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, batch_dim: int, dataset_size: int, device: torch.device):
        super().__init__()
        self.batch_dim = batch_dim
        assert self.batch_dim in [1, 2]
        self.dataset_size = dataset_size
        self.data = torch.tensor([], device=device)

    def __len__(self) -> int:
        return prod(self.data.shape[: self.batch_dim])

    def __getitem__(
        self, idx: torch.Tensor | tuple[int | slice | torch.Tensor, int | slice | torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(idx, tuple):
            assert self.batch_dim == len(idx)
        return self.data[idx]

    def add(self, new_batch: torch.Tensor) -> None:
        self.data = torch.cat([self.data, new_batch.detach()], dim=0)
        self.trim_if_needed()

    def trim_if_needed(self) -> None:
        if len(self) > self.dataset_size:
            if self.batch_dim == 1:
                trim_from = self.data.shape[0] - self.dataset_size
            else:  # batch_dim == 2
                trim_from = self.data.shape[0] - self.dataset_size // self.data.shape[1]
            self.data = self.data[trim_from:]  # FIFO

    def update(
        self,
        idx: torch.Tensor | tuple[int | slice | torch.Tensor, int | slice | torch.Tensor],
        new_batch: torch.Tensor,
    ) -> None:
        if isinstance(idx, tuple):
            assert self.batch_dim == len(idx)
        self.data[idx] = new_batch.detach()

    def reorder(
        self, idx: torch.Tensor | tuple[int | slice | torch.Tensor, int | slice | torch.Tensor]
    ) -> None:
        if isinstance(idx, tuple):
            assert self.batch_dim == len(idx)
        self.data = self.data[idx]
