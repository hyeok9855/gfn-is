from typing import Callable
from functools import partial

import torch


def get_discretizer(discretizer: str, T: int, *args, **kwargs) -> Callable[[int], torch.Tensor]:
    if discretizer == "uniform":
        return partial(uniform_discretizer, T=T)
    elif discretizer == "random":
        return partial(random_discretizer, T=T, *args, **kwargs)
    elif discretizer == "equidistant":
        return partial(shifted_equidistant_discretizer, T=T, *args, **kwargs)
    else:
        raise ValueError(f"Unknown discretizer: {discretizer}")


def uniform_discretizer(batch_size: int, T: int, *args, **kwargs) -> torch.Tensor:
    ts = torch.linspace(0, 1, T + 1).repeat(batch_size, 1)
    return ts


def random_discretizer(batch_size: int, T: int, max_ratio=10.0, *args, **kwargs) -> torch.Tensor:
    ts = (torch.rand(batch_size, T) * (max_ratio - 1) + 1).cumsum(1)
    ts = torch.cat([torch.zeros(batch_size, 1), ts], 1) / ts[:, -1].unsqueeze(1)
    return ts


def shifted_equidistant_discretizer(batch_size: int, T: int, eps=1e-4, *args, **kwargs) -> torch.Tensor:
    bound = 1 / T - eps
    noise = torch.empty(batch_size, 1).uniform_(- bound, bound)
    steps = (torch.arange(1, T) / T).unsqueeze(0) + noise
    ts = torch.cat([torch.zeros(batch_size, 1), steps, torch.ones(batch_size, 1)], dim=1)
    return ts
