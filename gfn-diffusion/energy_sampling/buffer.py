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


class ReplayBuffer:
    def __init__(
        self,
        buffer_size,
        device: torch.device,
        prioritization: str,
        rank_k: float = 0.01,
        logr_lb: float | None = None,
    ):
        assert prioritization in ["none", "reward", "loss", "delta"]
        self.buffer_size = buffer_size
        self.device = device
        self.prioritization = prioritization
        self.logr_lb = logr_lb
        self.rank_k = rank_k

        self.x_dataset = CustomDataset(self.device, buffer_size)
        self.logr_dataset = CustomDataset(self.device, buffer_size)
        self.delta_dataset = CustomDataset(self.device, buffer_size)  # Note: delta ** 2 = loss
        self.dataset = ZipDataset(self.x_dataset, self.logr_dataset, self.delta_dataset)
        self.prioritized_indices: torch.Tensor | None = None

    def __len__(self):
        return len(self.dataset)


    def _add_or_update(
        self,
        action: str,
        xs: torch.Tensor,
        log_rewards: torch.Tensor,
        deltas: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> None:
        if action == "update":
            assert indices is not None

        deltas = deltas if deltas is not None else torch.zeros_like(log_rewards, device=log_rewards.device)
        zipped = zip(
            [self.x_dataset, self.logr_dataset, self.delta_dataset],
            [xs, log_rewards, deltas],
        )

        for _ds, _data in zipped:
            if _ds is not None:
                assert _data is not None
                if action == "add":
                    _ds.add(_data)
                elif action == "update":
                    _ds.update(indices, _data)

        self.reset()

    def add(
        self,
        xs: torch.Tensor,
        log_rewards: torch.Tensor,
        deltas: torch.Tensor | None = None,
    ) -> None:
        # filter out the outliers in the log-rewards for numerical stability
        if self.logr_lb is not None:
            mask = log_rewards > self.logr_lb
            xs = xs[mask]
            log_rewards = log_rewards[mask]
            deltas = deltas[mask] if deltas is not None else None

        self._add_or_update("add", xs, log_rewards, deltas)

    def update(
        self,
        indices: torch.Tensor,
        xs: torch.Tensor,
        log_rewards: torch.Tensor,
        deltas: torch.Tensor | None = None,
    ) -> None:
        self._add_or_update("update", xs, log_rewards, deltas, indices)

    def reset(self) -> None:
        self.dataset = ZipDataset(self.x_dataset, self.logr_dataset, self.delta_dataset)

    def sample(
        self, batch_size: int, prioritized=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = torch.ones(len(self.dataset), device=self.device)
        if prioritized and self.prioritization != "none":
            match self.prioritization:
                case "reward":
                    scores = self.logr_dataset.data
                case "loss":
                    raise NotImplementedError
                    scores = self.delta_dataset.data**2
                case "delta":
                    scores = self.delta_dataset.data
                case _:
                    raise NotImplementedError

            ranks = torch.argsort(torch.argsort(-scores))
            weights = 1.0 / (self.rank_k * len(scores) + ranks)

        indices = torch.multinomial(weights, batch_size, replacement=False)
        xs, log_rewards, deltas = self.dataset[indices]

        return xs, log_rewards, deltas, indices
