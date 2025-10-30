from abc import ABC, abstractmethod
from typing import Callable, Literal

import torch

from buffers.datasets import CustomDataset
from utils.train_utils import binary_search_smoothing


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

    @property
    def all_datasets(self) -> list[CustomDataset]:
        all_datasets = []
        for attr in dir(self):
            if isinstance(getattr(self, attr), CustomDataset):
                all_datasets.append(getattr(self, attr))
        return all_datasets

    def __len__(self) -> int:
        return len(self.states_dataset)

    def get_logr_mask(self, log_fs: torch.Tensor) -> torch.Tensor:
        if self.logr_lb is None:
            mask = torch.ones_like(log_fs, dtype=torch.bool)
        else:
            if log_fs.ndim == 1:
                mask = log_fs > self.logr_lb
            elif log_fs.ndim == 2:
                mask = log_fs[:, -1] > self.logr_lb
            else:
                raise ValueError(f"log_fs has {log_fs.ndim} dimensions, expected 1 or 2")
        return mask

    def add(self, **data_dict: torch.Tensor) -> None:
        mask = self.get_logr_mask(data_dict["log_fs"])
        for key, data in data_dict.items():
            dataset = getattr(self, f"{key}_dataset")
            assert isinstance(dataset, CustomDataset)
            dataset.add(data[mask])

    def update(self, indices: torch.Tensor, **data_dict: torch.Tensor) -> None:
        for key, data in data_dict.items():
            dataset = getattr(self, f"{key}_dataset")
            assert isinstance(dataset, CustomDataset)
            dataset.update(indices, data)

    def discard(self, indices: torch.Tensor) -> None:
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

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(self) > 0, "Buffer is empty"

        weights = torch.ones(len(self.states_dataset), device=self.device)
        match self.prioritization:
            case "none":
                pass
            case "target":
                weights = self.log_fs_dataset.data  # log_fs is log_rs, (bs,)
                weights = weights.softmax(dim=0)
            case "loss":
                assert self.losses_dataset is not None
                weights = self.losses_dataset.data  # (bs,)
            case "normalized_iw":
                assert self.normalized_iws_dataset is not None
                weights = self.normalized_iws_dataset.data  # (bs,)
            case "iw":
                raise ValueError("Use PIWTerminalStateBuffer for iw-based prioritization")
            case _:
                raise NotImplementedError

        replacement = True if self.prioritization == "normalized_iw" else False
        indices = self.sampling_func(weights, batch_size, replacement)
        states, log_fs = self.states_dataset[indices], self.log_fs_dataset[indices]

        return states, log_fs, indices

    def sample_terminal(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs, log_rs, _ = self.sample(batch_size)
        return xs, log_rs


class IntermediateStateBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        device: torch.device,
        prioritization: Literal["none", "target", "iw", "normalized_iw"],
        sampling_func: Callable[[torch.Tensor, int, bool], torch.Tensor],
        logr_lb: float | None = None,
        **kwargs,
    ) -> None:
        assert prioritization in ["none", "target", "iw", "normalized_iw"]
        batch_dim = 2  # (bs, n_timesteps)
        super().__init__(
            batch_dim=batch_dim,
            buffer_size=buffer_size,
            device=device,
            prioritization=prioritization,
            sampling_func=sampling_func,
            logr_lb=logr_lb,
        )
        self.ts_dataset = CustomDataset(batch_dim, buffer_size, device)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(self) > 0, "Buffer is empty"
        assert self.ts_dataset is not None

        len1, len2 = self.log_fs_dataset.data.shape

        weights = torch.ones(len1, len2, device=self.device)
        match self.prioritization:
            case "none":
                pass
            case "target":
                weights = self.log_fs_dataset.data  # (len1, len2)
                weights = weights.softmax(dim=0)
            case "normalized_iw":  # Already normalized for each timesteps
                assert self.normalized_iws_dataset is not None
                weights = self.normalized_iws_dataset.data  # (len1, len2)
            case _:
                raise NotImplementedError

        # TODO: Sample timestep first, and then sample state
        replacement = True if self.prioritization == "normalized_iw" else False
        indices = self.sampling_func(weights.flatten(), batch_size, replacement)
        dim1_indices = indices // len2  # (bs,)
        dim2_indices = indices % len2  # (bs,)

        states = self.states_dataset[dim1_indices, dim2_indices]
        ts = self.ts_dataset[dim1_indices, dim2_indices]
        log_fs = self.log_fs_dataset[dim1_indices, dim2_indices]

        return states, ts, log_fs, indices

    def sample_timestep(
        self, batch_size: int, t_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(self) > 0, "Buffer is empty"
        assert self.ts_dataset is not None
        assert t_idx < self.ts_dataset.data.shape[1]

        t_states = self.states_dataset[:, t_idx]
        t_ts = self.ts_dataset[:, t_idx]
        t_log_fs = self.log_fs_dataset[:, t_idx]

        weights = torch.ones(len(t_states), device=self.device)
        match self.prioritization:
            case "none":
                pass
            case "target":
                weights = t_log_fs
                weights = weights.softmax(dim=0)
            case "normalized_iw":
                assert self.normalized_iws_dataset is not None
                weights = self.normalized_iws_dataset[:, t_idx]
            case "iw":
                raise ValueError("Use PIWIntermediateStateBuffer for iw-based prioritization")
            case _:
                raise NotImplementedError

        replacement = True if self.prioritization == "normalized_iw" else False
        indices = self.sampling_func(weights, batch_size, replacement)
        xs, ts, log_fs = t_states[indices], t_ts[indices], t_log_fs[indices]
        return xs, ts, log_fs

    def sample_terminal(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.ts_dataset is not None
        xs, _, log_rs = self.sample_timestep(batch_size, self.ts_dataset.data.shape[1] - 1)
        return xs, log_rs


##### PIW BUFFER #####
class PIWTerminalStateBuffer(TerminalStateBuffer):
    def __init__(
        self,
        buffer_size: int,
        device: torch.device,
        sampling_func: Callable[[torch.Tensor, int, bool], torch.Tensor],
        logr_lb: float | None = None,
        smoothing_strategy: str = "temper",
        target_ess: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(
            batch_dim=1,  # (bs,)
            buffer_size=buffer_size,
            device=device,
            prioritization="iw",
            sampling_func=sampling_func,
            logr_lb=logr_lb,
        )
        self.log_iws_dataset = CustomDataset(1, buffer_size, device)
        self.batch_idx_dataset = CustomDataset(1, buffer_size, device)
        self.batch_idx = 0

        # PIW-specific parameters
        assert 0.0 <= target_ess <= 1.0
        self.target_ess = target_ess
        self.smoothing_strategy = smoothing_strategy
        self.discard_strategy = "fifo"  # TODO: Implement GIS-based discard strategy

    def add(self, **data_dict: torch.Tensor) -> None:
        batch_idx_tensor = torch.zeros_like(data_dict["log_fs"], dtype=torch.int64) + self.batch_idx
        self.batch_idx += 1

        if (
            len(self) + len(data_dict["log_fs"]) > self.buffer_size
            and self.discard_strategy == "gis"
        ):
            # TODO: Implement discard strategy based on Group Importance Sampling (GIS)
            # May need to do something with the batch_idx_tensor
            assert len(self.log_fs_dataset) + len(data_dict["log_fs"]) < self.buffer_size
        else:  # discard == "fifo"
            pass  # This is handled in CustomDataset

        super().add(**data_dict, batch_idx=batch_idx_tensor)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(self) > 0, "Buffer is empty"

        assert self.prioritization == "iw"
        log_iws = self.log_iws_dataset.data  # (bs,)

        # apply ESS-based smoothing
        log_iws_smoothed = binary_search_smoothing(
            log_weights=log_iws.unsqueeze(1),
            target_ess=int(self.target_ess * len(log_iws)),
            smoothing_strategy=self.smoothing_strategy,
        )
        weights = log_iws_smoothed.squeeze(1).softmax(dim=0)

        indices = self.sampling_func(weights, batch_size, True)
        states, log_fs = self.states_dataset[indices], self.log_fs_dataset[indices]
        return states, log_fs, indices

    def sample_terminal(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs, log_rs, _ = self.sample(batch_size)
        return xs, log_rs


class PIWIntermediateStateBuffer(IntermediateStateBuffer):
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError
