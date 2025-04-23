from abc import ABC, abstractmethod
from typing import Literal

import torch

from buffers.datasets import CustomDataset


class BaseBuffer(ABC):
    def __init__(
        self,
        batch_dim: int,
        buffer_size: int,
        device: torch.device,
        prioritization: Literal["none", "target", "loss", "normalized_iw"] = "none",
        sampling_strategy: Literal["proportional", "rank"] = "proportional",
        rank_k: float = 0.01,
        logr_lb: float | None = None,
    ) -> None:
        assert prioritization in ["none", "reward", "loss", "normalized_iw"]
        if prioritization == "normalized_iw":
            assert sampling_strategy == "proportional"

        self.buffer_size = buffer_size
        self.device = device
        self.prioritization = prioritization
        self.sampling_strategy = sampling_strategy
        self.rank_k = rank_k
        self.logr_lb = logr_lb

        self.states_dataset = CustomDataset(batch_dim, buffer_size, device)
        self.log_fs_dataset = CustomDataset(batch_dim, buffer_size, device)
        self.losses_dataset = (
            CustomDataset(batch_dim, buffer_size, device) if prioritization == "loss" else None
        )
        self.normalized_iws_dataset = (
            CustomDataset(batch_dim, buffer_size, device)
            if prioritization == "normalized_iw"
            else None
        )
        self.ts_dataset: CustomDataset | None = None

    def __len__(self) -> int:
        return len(self.states_dataset)

    def add(
        self,
        states: torch.Tensor,
        log_fs: torch.Tensor,
        losses: torch.Tensor | None = None,
        normalized_iws: torch.Tensor | None = None,
        ts: torch.Tensor | None = None,
    ) -> None:
        for data, dataset in zip(
            [states, log_fs, losses, normalized_iws, ts],
            [
                self.states_dataset,
                self.log_fs_dataset,
                self.losses_dataset,
                self.normalized_iws_dataset,
                self.ts_dataset,
            ],
        ):
            if dataset is not None:
                assert data is not None
                dataset.add(data)

    def update(
        self,
        indices: torch.Tensor,
        states: torch.Tensor | None = None,
        log_fs: torch.Tensor | None = None,
        losses: torch.Tensor | None = None,
        normalized_iws: torch.Tensor | None = None,
        ts: torch.Tensor | None = None,
    ) -> None:
        for data, dataset in zip(
            [states, log_fs, losses, normalized_iws, ts],
            [
                self.states_dataset,
                self.log_fs_dataset,
                self.losses_dataset,
                self.normalized_iws_dataset,
                self.ts_dataset,
            ],
        ):
            if data is not None:
                assert dataset is not None
                dataset.update(indices, data)

    @abstractmethod
    def sample(self, batch_size: int, prioritized=True) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @abstractmethod
    def sample_terminal(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class TerminalStateBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size,
        device: torch.device,
        prioritization: Literal["none", "target", "loss", "normalized_iw"] = "none",
        sampling_strategy: Literal["proportional", "rank"] = "proportional",
        rank_k: float = 0.01,
        logr_lb: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            batch_dim=1,  # (bs,)
            buffer_size=buffer_size,
            device=device,
            prioritization=prioritization,
            sampling_strategy=sampling_strategy,
            rank_k=rank_k,
            logr_lb=logr_lb,
        )

    def add(
        self,
        states: torch.Tensor,
        log_fs: torch.Tensor,
        losses: torch.Tensor | None = None,
        normalized_iws: torch.Tensor | None = None,
        ts: torch.Tensor | None = None,
    ) -> None:
        assert ts is None  # useless
        # Convert all tensor from (bs, T) to (bs,)
        states = states[:, -1]  # (bs,)
        log_fs = log_fs[:, -1]  # (bs,)
        losses = losses[:, -1] if losses is not None else None  # (bs,)
        normalized_iws = normalized_iws[:, -1] if normalized_iws is not None else None  # (bs,)

        # filter out the outliers in the log-rewards for numerical stability
        if self.logr_lb is not None:
            mask = log_fs > self.logr_lb  # log_fs is log_rs
            states = states[mask]
            log_fs = log_fs[mask]
            losses = losses[mask] if losses is not None else None
            normalized_iws = normalized_iws[mask] if normalized_iws is not None else None

        super().add(states, log_fs, losses, normalized_iws, ts)

    def sample(
        self, batch_size: int, prioritized=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(self) > 0, "Buffer is empty"

        weights = torch.ones(len(self.states_dataset), device=self.device)
        if prioritized and self.prioritization != "none":
            match self.prioritization:
                case "target":
                    scores = self.log_fs_dataset.data  # log_fs is log_rs, (bs,)
                case "loss":
                    assert self.losses_dataset is not None
                    scores = self.losses_dataset.data  # (bs,)
                case "normalized_iw":
                    assert self.normalized_iws_dataset is not None
                    scores = self.normalized_iws_dataset.data  # (bs,)
                case _:
                    raise NotImplementedError

            if self.sampling_strategy == "proportional":
                if self.prioritization in ["loss", "normalized_iw"]:
                    weights = scores
                else:  # target
                    weights = torch.exp(scores - torch.max(scores))
            elif self.sampling_strategy == "rank":
                assert self.prioritization != "normalized_iw"
                ranks = torch.argsort(torch.argsort(-scores))
                weights = 1.0 / (self.rank_k * len(scores) + ranks)
            else:
                raise NotImplementedError

        replacement = True if self.prioritization == "normalized_iw" else False
        indices = torch.multinomial(weights, batch_size, replacement=replacement)
        states, log_fs = self.states_dataset[indices], self.log_fs_dataset[indices]

        return states, log_fs, indices

    def sample_terminal(
        self, batch_size: int, prioritized=True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs, log_rs, _ = self.sample(batch_size, prioritized)
        return xs, log_rs


class IntermediateStateBuffer(BaseBuffer):
    def __init__(self):
        raise NotImplementedError
