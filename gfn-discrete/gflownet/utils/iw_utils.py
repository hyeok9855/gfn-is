from typing import Callable

import torch


def ess(
    log_weights: torch.Tensor | None = None,  # (bs, T)
    normalized_weights: torch.Tensor | None = None,  # (bs, T)
) -> torch.Tensor:
    if normalized_weights is None:
        assert log_weights is not None
        normalized_weights = log_weights.softmax(dim=0)  # (bs, T)
    return 1 / (normalized_weights**2).sum(dim=0)  # (T,)


def binary_search_smoothing(
    log_weights: torch.Tensor,  # (bs, T)
    target_ess: float,
    smoothing_strategy: str,
    tol=1e-3,
    max_steps=1000,
) -> torch.Tensor:
    func = get_smoothing_func(smoothing_strategy)

    search_min, search_max = get_min_max(func, log_weights)
    search_min = torch.tensor(search_min, device=log_weights.device).repeat(1, log_weights.shape[1])
    search_max = torch.tensor(search_max, device=log_weights.device).repeat(1, log_weights.shape[1])
    mid = (search_min + search_max) / 2  # (1, T)
    original_order = ess(func(log_weights, search_min)) < ess(func(log_weights, search_max))

    dones = ess(log_weights=log_weights) >= target_ess  # (T,)
    log_weights_smoothed = log_weights.clone()  # (bs, T)

    steps = 0
    while not dones.all():
        steps += 1
        mid[0, ~dones] = (search_min[0, ~dones] + search_max[0, ~dones]) / 2  # (1, T)

        new_log_weights = func(log_weights, mid)  # (bs, T)
        new_ess = ess(log_weights=new_log_weights)  # (T,)
        new_dones = (~dones) & (abs(new_ess - target_ess) / target_ess < tol)  # (T,)
        log_weights_smoothed[:, new_dones] = new_log_weights[:, new_dones]
        dones = dones | new_dones

        search_max = torch.where((new_ess > target_ess) == original_order, mid, search_max)
        search_min = torch.where((new_ess < target_ess) == original_order, mid, search_min)

        if steps > max_steps:
            print(f"Warning: Binary search failed in {max_steps} steps")
            log_weights_smoothed[:, ~dones] = new_log_weights[:, ~dones]
            break
    return log_weights_smoothed


def clip_below(log_weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return log_weights.clamp(min=value)


def clip_above(log_weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return log_weights.clamp(max=value)


def temper(log_weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return log_weights / value


def get_smoothing_func(
    smoothing_strategy: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if smoothing_strategy == "temper":
        return temper
    elif smoothing_strategy == "clip_above":
        return clip_above
    elif smoothing_strategy == "clip_below":
        return clip_below
    else:
        raise ValueError(f"Invalid smoothing strategy: {smoothing_strategy}")


def get_min_max(func: Callable, log_weights: torch.Tensor) -> tuple[float, float]:
    _min = torch.nan_to_num(log_weights, nan=float("inf"), neginf=float("inf")).min().item()
    _max = torch.nan_to_num(log_weights, nan=float("-inf"), posinf=float("-inf")).max().item()
    if func == clip_above or func == clip_below:
        return _min, _max
    elif func == temper:
        return 1.0, (_max - _min) / 2
    else:
        raise ValueError(f"Invalid function: {func}")
