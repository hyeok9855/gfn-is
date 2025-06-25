from abc import ABC, abstractmethod
from math import prod

import torch
import numpy as np
from typing import Callable, Literal, cast

from gflownet.utils.iw_utils import binary_search_smoothing


class CustomDataset:
    def __init__(self, batch_dim: int, dataset_size: int) -> None:
        super().__init__()
        self.batch_dim = batch_dim
        assert self.batch_dim in [1, 2]
        self.dataset_size = dataset_size
        self.data: torch.Tensor | np.ndarray | None = None

    def __len__(self) -> int:
        assert self.data is not None
        return prod(self.data.shape[: self.batch_dim])

    def __getitem__(
        self, idx: np.ndarray | tuple[int | slice | np.ndarray, int | slice | np.ndarray]
    ) -> torch.Tensor | np.ndarray:
        if isinstance(idx, tuple):
            assert self.batch_dim == len(idx)
        assert self.data is not None
        return self.data[idx]

    def add(self, new_batch: torch.Tensor | np.ndarray) -> None:
        if self.data is None:
            self.data = new_batch
        else:
            if isinstance(self.data, torch.Tensor):
                self.data = torch.cat([self.data, new_batch.detach()], dim=0)  # type: ignore
            elif isinstance(self.data, np.ndarray):
                self.data = np.concatenate([self.data, new_batch], axis=0)
            else:
                raise ValueError(f"Unknown data type: {type(self.data)}")
            self.trim_if_needed()

    def trim_if_needed(self) -> None:
        assert self.data is not None
        if len(self) > self.dataset_size:
            if self.batch_dim == 1:
                trim_from = self.data.shape[0] - self.dataset_size
            else:  # batch_dim == 2
                trim_from = self.data.shape[0] - self.dataset_size // self.data.shape[1]
            self.data = self.data[trim_from:]  # FIFO

    def update(
        self,
        indices: np.ndarray | tuple[int | slice | np.ndarray, int | slice | np.ndarray],
        new_batch: torch.Tensor | np.ndarray,
    ) -> None:
        if isinstance(indices, tuple):
            assert self.batch_dim == len(indices)
        if isinstance(self.data, torch.Tensor):
            self.data[indices] = new_batch.detach()  # type: ignore
        elif isinstance(self.data, np.ndarray):
            self.data[indices] = new_batch
        else:
            raise ValueError(f"Unknown data type: {type(self.data)}")

    def discard(self, indices: np.ndarray) -> None:
        assert indices.ndim == 1
        assert self.data is not None
        self.data = self.data[~indices]  # type: ignore


class BaseBuffer(ABC):
    def __init__(
        self,
        batch_dim: int,
        buffer_size: int,
        device: torch.device,
        prioritization: Literal["none", "target", "loss", "iw", "normalized_iw"],
        sampling_func: Callable[[torch.Tensor, int, bool], torch.Tensor],
        logr_lb: float | None = None,
    ) -> None:
        assert prioritization in ["none", "target", "loss", "iw", "normalized_iw"]
        if prioritization == "normalized_iw":
            assert sampling_func is not None

        self.buffer_size = buffer_size
        self.device = device
        self.prioritization = prioritization
        self.sampling_func = sampling_func
        self.logr_lb = logr_lb

        self.states_dataset = CustomDataset(batch_dim, buffer_size)
        self.log_fs_dataset = CustomDataset(batch_dim, buffer_size)
        self.losses_dataset = (
            CustomDataset(batch_dim, buffer_size) if prioritization == "loss" else None
        )
        self.log_iws_dataset = CustomDataset(1, buffer_size) if prioritization == "iw" else None
        self.normalized_iws_dataset = (
            CustomDataset(batch_dim, buffer_size)
            if prioritization == "normalized_iw"
            else None
        )

    @property
    def all_datasets(self) -> list[CustomDataset]:
        all_datasets = []
        for attr in dir(self):
            if isinstance(getattr(self, attr), CustomDataset):
                all_datasets.append(getattr(self, attr))
        return all_datasets

    def __len__(self) -> int:
        return len(self.states_dataset)

    def get_logr_mask(self, log_fs: torch.Tensor) -> np.ndarray:
        if self.logr_lb is None:
            mask = torch.ones_like(log_fs, dtype=torch.bool)
        else:
            if log_fs.ndim == 1:
                mask = log_fs > self.logr_lb
            elif log_fs.ndim == 2:
                mask = log_fs[:, -1] > self.logr_lb
            else:
                raise ValueError(f"log_fs has {log_fs.ndim} dimensions, expected 1 or 2")
        return mask.detach().cpu().numpy()

    def add(self, **data_dict: torch.Tensor | np.ndarray) -> None:
        assert isinstance(data_dict["log_fs"], torch.Tensor)
        mask = self.get_logr_mask(data_dict["log_fs"])
        for key, data in data_dict.items():
            dataset = getattr(self, f"{key}_dataset")
            assert isinstance(dataset, CustomDataset)
            dataset.add(data[mask])

    def update(self, indices: np.ndarray, **data_dict: torch.Tensor | np.ndarray) -> None:
        for key, data in data_dict.items():
            dataset = getattr(self, f"{key}_dataset")
            assert isinstance(dataset, CustomDataset)
            dataset.update(indices, data)

    def discard(self, indices: np.ndarray) -> None:
        for dataset in self.all_datasets:
            dataset.discard(indices)

    @abstractmethod
    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @abstractmethod
    def sample_terminal(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class TerminalStateBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        device: torch.device,
        prioritization: Literal["none", "target", "loss", "iw", "normalized_iw"],
        sampling_func: Callable[[torch.Tensor, int, bool], torch.Tensor],
        logr_lb: float | None = None,
        target_ess: float = 0.05,
        smoothing_strategy: str = "temper",
        **kwargs,
    ) -> None:
        super().__init__(
            batch_dim=1,  # (bs,)
            buffer_size=buffer_size,
            device=device,
            prioritization=prioritization,
            sampling_func=sampling_func,
            logr_lb=logr_lb,
        )

        # For prioritization == "iw"
        assert 0.0 <= target_ess <= 1.0
        self.target_ess = target_ess
        self.smoothing_strategy = smoothing_strategy


    def sample(self, batch_size: int) -> tuple[torch.Tensor | np.ndarray, torch.Tensor, np.ndarray]:
        assert len(self) > 0, "Buffer is empty"

        weights = torch.ones(len(self.states_dataset), device=self.device)
        match self.prioritization:
            case "none":
                pass
            case "target":
                weights = cast(torch.Tensor, self.log_fs_dataset.data)  # log_fs is log_rs, (bs,)
                weights = weights.softmax(dim=0)
            case "loss":
                assert self.losses_dataset is not None
                weights = cast(torch.Tensor, self.losses_dataset.data)  # (bs,)
            case "iw":
                assert self.log_iws_dataset is not None
                log_iws = cast(torch.Tensor, self.log_iws_dataset.data)  # (bs,)
                # apply ESS-based smoothing
                log_iws_smoothed = binary_search_smoothing(
                    log_weights=log_iws.unsqueeze(1),
                    target_ess=int(self.target_ess * len(log_iws)),
                    smoothing_strategy=self.smoothing_strategy,
                )
                weights = log_iws_smoothed.squeeze(1).softmax(dim=0)
            case "normalized_iw":
                assert self.normalized_iws_dataset is not None
                weights = cast(torch.Tensor, self.normalized_iws_dataset.data)  # (bs,)
            case _:
                raise NotImplementedError

        replacement = True if self.prioritization in ["iw", "normalized_iw"] else False
        indices = self.sampling_func(weights, batch_size, replacement)
        indices = indices.detach().cpu().numpy()
        states, log_fs = self.states_dataset[indices], self.log_fs_dataset[indices]

        return states, cast(torch.Tensor, log_fs), indices

    def sample_terminal(self, batch_size: int) -> tuple[torch.Tensor | np.ndarray, torch.Tensor]:
        xs, log_rs, _ = self.sample(batch_size)
        return xs, log_rs
