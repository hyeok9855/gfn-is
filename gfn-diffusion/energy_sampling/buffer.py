from typing import Literal
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
        prioritization: Literal["none", "reward", "loss", "normalized_iw"] = "none",
        sampling_strategy: Literal["proportional", "rank"] = "proportional",
        rank_k: float = 0.01,
        logr_lb: float | None = None,
    ):
        assert prioritization in ["none", "reward", "loss", "normalized_iw"]
        if prioritization == "normalized_iw":
            assert sampling_strategy == "proportional"

        self.buffer_size = buffer_size
        self.device = device
        self.prioritization = prioritization
        self.sampling_strategy = sampling_strategy
        self.rank_k = rank_k
        self.logr_lb = logr_lb

        self.x_dataset = CustomDataset(self.device, buffer_size)
        self.log_r_dataset = CustomDataset(self.device, buffer_size)
        self.loss_dataset = CustomDataset(self.device, buffer_size)
        self.log_iw_dataset = CustomDataset(self.device, buffer_size)
        self.normalized_iw_dataset = CustomDataset(self.device, buffer_size)
        self.dataset = ZipDataset(
            self.x_dataset,
            self.log_r_dataset,
            self.loss_dataset,
            self.log_iw_dataset,
            self.normalized_iw_dataset,
        )

    def __len__(self):
        return len(self.dataset)

    def _add_or_update(
        self,
        action: str,
        xs: torch.Tensor,
        log_rewards: torch.Tensor,
        losses: torch.Tensor | None = None,
        log_iws: torch.Tensor | None = None,
        normalized_iws: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> None:
        if action == "update":
            assert indices is not None

        losses = (
            losses
            if losses is not None
            else torch.zeros_like(log_rewards, device=log_rewards.device)
        )
        log_iws = (
            log_iws
            if log_iws is not None
            else torch.zeros_like(log_rewards, device=log_rewards.device)
        )
        normalized_iws = (
            normalized_iws
            if normalized_iws is not None
            else torch.zeros_like(log_rewards, device=log_rewards.device)
        )

        zipped = zip(
            [
                self.x_dataset,
                self.log_r_dataset,
                self.loss_dataset,
                self.log_iw_dataset,
                self.normalized_iw_dataset,
            ],
            [xs, log_rewards, losses, log_iws, normalized_iws],
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
        losses: torch.Tensor | None = None,
        log_iws: torch.Tensor | None = None,
        normalized_iws: torch.Tensor | None = None,
    ) -> None:
        # filter out the outliers in the log-rewards for numerical stability
        if self.logr_lb is not None:
            mask = log_rewards > self.logr_lb
            xs = xs[mask]
            log_rewards = log_rewards[mask]
            losses = losses[mask] if losses is not None else None
            log_iws = log_iws[mask] if log_iws is not None else None
            normalized_iws = normalized_iws[mask] if normalized_iws is not None else None

        self._add_or_update("add", xs, log_rewards, losses, log_iws, normalized_iws)

    def update(
        self,
        indices: torch.Tensor,
        xs: torch.Tensor,
        log_rewards: torch.Tensor,
        losses: torch.Tensor | None = None,
        log_iws: torch.Tensor | None = None,
        normalized_iws: torch.Tensor | None = None,
    ) -> None:
        self._add_or_update("update", xs, log_rewards, losses, log_iws, normalized_iws, indices)

    def reset(self) -> None:
        self.dataset = ZipDataset(
            self.x_dataset,
            self.log_r_dataset,
            self.loss_dataset,
            self.log_iw_dataset,
            self.normalized_iw_dataset,
        )

    def sample(
        self, batch_size: int, prioritized=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = torch.ones(len(self.dataset), device=self.device)
        if prioritized and self.prioritization != "none":
            match self.prioritization:
                case "reward":
                    scores = self.log_r_dataset.data
                case "loss":
                    scores = self.loss_dataset.data
                case "normalized_iw":
                    scores = self.normalized_iw_dataset.data
                case _:
                    raise NotImplementedError

            if self.sampling_strategy == "proportional":
                if self.prioritization in ["loss", "normalized_iw"]:
                    weights = scores
                else:  # reward
                    weights = torch.exp(scores - torch.max(scores))
            elif self.sampling_strategy == "rank":
                assert self.prioritization != "normalized_iw"
                ranks = torch.argsort(torch.argsort(-scores))
                weights = 1.0 / (self.rank_k * len(scores) + ranks)
            else:
                raise NotImplementedError

        replacement = True if self.prioritization == "normalized_iw" else False
        indices = torch.multinomial(weights, batch_size, replacement=replacement)
        xs, log_rewards, _, log_iws, _ = self.dataset[indices]

        return xs, log_rewards, log_iws, indices
