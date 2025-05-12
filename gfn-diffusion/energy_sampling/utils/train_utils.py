from typing import Callable

import torch

from models import GFN


def get_gfn_optimizer(
    gfn_model: GFN,
    lr_fwd: float,
    lr_bwd: float,
    lr_flow: float,
    lr_beta: float,
    lr_lgv: float,
    use_weight_decay=False,
    weight_decay=1e-7,
    use_scheduler=False,
    milestones: list[int] = [100000],
    gamma: float = 1.0,
):

    module_param_groups = gfn_model.pred_module.get_param_groups()

    param_groups = []
    param_groups.append({"params": module_param_groups.forward_params, "lr": lr_fwd})
    param_groups.append({"params": module_param_groups.backward_params, "lr": lr_bwd})
    param_groups.append({"params": module_param_groups.flow_params, "lr": lr_flow})
    param_groups.append({"params": module_param_groups.lgv_params, "lr": lr_lgv})

    if gfn_model.beta_model is not None:
        param_groups.append({"params": gfn_model.beta_model, "lr": lr_beta})

    gfn_optimizer = torch.optim.Adam(
        param_groups, lr=0.0, weight_decay=weight_decay if use_weight_decay else 0.0
    )

    gfn_scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(gfn_optimizer, milestones=milestones, gamma=gamma)
        if use_scheduler
        else None
    )
    return gfn_optimizer, gfn_scheduler


###########################################
### Importance weight related functions ###
###########################################


def get_normalized_weights(
    batch_size: int,
    ts: torch.Tensor,  # shape: (bs, T)
    log_fs: torch.Tensor,  # shape: (bs, T + 1)
    log_pbs: torch.Tensor,  # shape: (bs, T)
    log_pfs_exp: torch.Tensor,  # shape: (bs, T)
    device: torch.device,
    aux_target: str = "target",
    loss_type: str = "tb",
    target_ess: float = 0.0,
    smoothing: str = "temper",
    losses: torch.Tensor | None = None,
) -> torch.Tensor:
    # Compute importance weights
    log_iws_0t = torch.zeros(batch_size, ts.shape[1] - 1).to(device)  # shape: (bs, T)
    aux_target_measure_0t = torch.zeros_like(log_iws_0t)
    match aux_target:
        case "target":  # r(x)p_B(\tau|x)
            aux_target_measure_0t = log_fs[:, 1:] + log_pbs.cumsum(-1)
        case "loss":
            assert loss_type in ["tb", "logvar"] and losses is not None
            aux_target_measure_0t[:, -1] = losses.log()  # (bs,)
        case _:
            raise ValueError(f"Invalid aux_target: {aux_target}")
    proposal_measure_0t = log_pfs_exp.cumsum(-1)
    log_iws_0t = (aux_target_measure_0t - proposal_measure_0t).detach()

    # Importance weight smoothing
    normalized_iws_0t = log_iws_0t.softmax(dim=0)  # (bs, T)
    if target_ess != 0.0:
        target_ess = target_ess * batch_size if 0.0 <= target_ess <= 1.0 else target_ess
        assert 1.0 < target_ess <= batch_size, f"Invalid target ESS: {target_ess}"

        if (ess(normalized_weights=normalized_iws_0t) < target_ess).any():
            if smoothing == "mix_with_uniform":
                raise NotImplementedError("TODO: Implement this")
                # mixing_ratio = solve_mixing_ratio(normalized_iws_0t, target_ess=target_ess)
                # normalized_iws_0t = (
                #     1 - mixing_ratio
                # ) * normalized_iws_0t + mixing_ratio / batch_size
            else:  # binary search
                log_iws_0t = smoothing_with_binary_search(
                    log_iws_0t, target_ess, get_func(smoothing)
                )
                normalized_iws_0t = log_iws_0t.softmax(dim=0)
    return normalized_iws_0t


def solve_mixing_ratio(normalized_weights: torch.Tensor, target_ess: float) -> float:
    """
    Find the mixing ratio to achieve the target effective sample size (ESS)

    normalized_weights_mix = (1 - mixing_ratio) * normalized_weights + mixing_ratio / batch_size

    ESS_mix = 1 / (normalized_weights_mix^2).sum()
            = 1 / (((1 - mixing_ratio) * normalized_weights + mixing_ratio / batch_size)^2).sum()

    Solve the following equation for mixing_ratio:
    1 / ESS_mix = (((1 - mixing_ratio) * normalized_weights + mixing_ratio / batch_size)^2).sum()
                = 1 / target_ess

    This is equivalent to the following quadratic equation:
        A * (mixing_ratio^2) - 2 * B * mixing_ratio + C = 0
    where
        A = normalized_weights^2.sum() - 2 * normalized_weights.sum() / N + 1 / N
        B = normalized_weights^2.sum() - normalized_weights.sum() / N
        C = normalized_weights^2.sum() - 1 / target_ess
    """
    N = len(normalized_weights)
    nw_sum = 1.0
    nw_squared_sum = (normalized_weights**2).sum().item()

    ess_before = 1 / nw_squared_sum
    if ess_before >= target_ess:
        return 0.0

    A = nw_squared_sum - 2 * nw_sum / N + 1 / N
    B = nw_squared_sum - nw_sum / N
    C = nw_squared_sum - 1 / target_ess

    min_lhs = C - B**2 / A
    if min_lhs >= 0:
        raise ValueError(f"Cannot achieve target ESS: {target_ess}")
        return 1.0

    mixing_ratio_1 = (B + (B**2 - A * C) ** 0.5) / A
    mixing_ratio_2 = (B - (B**2 - A * C) ** 0.5) / A

    valid_1 = mixing_ratio_1 >= 0.0 and mixing_ratio_1 <= 1.0
    valid_2 = mixing_ratio_2 >= 0.0 and mixing_ratio_2 <= 1.0

    if valid_1 and valid_2:
        raise ValueError(f"Multiple solutions: {mixing_ratio_1}, {mixing_ratio_2}")
    elif valid_1:
        return mixing_ratio_1
    elif valid_2:
        return mixing_ratio_2
    else:
        raise ValueError(f"No valid solution: {mixing_ratio_1}, {mixing_ratio_2}")


def ess(
    log_weights: torch.Tensor | None = None,  # shape: (bs, T)
    normalized_weights: torch.Tensor | None = None,  # shape: (bs, T)
) -> torch.Tensor:
    if normalized_weights is None:
        assert log_weights is not None
        normalized_weights = log_weights.softmax(dim=0)  # shape: (bs, T)
    return 1 / (normalized_weights**2).sum(dim=0)  # shape: (T,)


def smoothing_with_binary_search(
    log_weights: torch.Tensor,  # shape: (bs, T)
    target_ess: float,
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    tol=1e-2,
    max_steps=1000,
) -> torch.Tensor:
    search_min, search_max = get_min_max(func, log_weights)
    search_min = torch.tensor(search_min, device=log_weights.device).repeat(1, log_weights.shape[1])
    search_max = torch.tensor(search_max, device=log_weights.device).repeat(1, log_weights.shape[1])
    mid = (search_min + search_max) / 2  # shape: (1, T)
    original_order = ess(func(log_weights, search_min)) < ess(func(log_weights, search_max))

    dones = ess(log_weights=log_weights) >= target_ess  # shape: (T,)
    log_weights_smoothed = log_weights.clone()  # shape: (bs, T)

    steps = 0
    while not dones.all():
        steps += 1
        mid[0, ~dones] = (search_min[0, ~dones] + search_max[0, ~dones]) / 2  # shape: (1, T)

        new_log_weights = func(log_weights, mid)  # shape: (bs, T)
        new_ess = ess(log_weights=new_log_weights)  # shape: (T,)
        new_dones = (~dones) & (abs(new_ess - target_ess) < tol)  # shape: (T,)
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


def get_func(smoothing: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if smoothing == "clip_below":
        return clip_below
    elif smoothing == "clip_above":
        return clip_above
    elif smoothing == "temper":
        return temper
    else:
        raise ValueError(f"Invalid function: {smoothing}")


def get_min_max(func: Callable, log_weights: torch.Tensor) -> tuple[float, float]:
    _min = torch.nan_to_num(log_weights, nan=float("inf"), neginf=float("inf")).min().item()
    _max = torch.nan_to_num(log_weights, nan=float("-inf"), posinf=float("-inf")).max().item()
    if func == clip_above or func == clip_below:
        return _min, _max
    elif func == temper:
        return 1.0, (_max - _min) / 2
    else:
        raise ValueError(f"Invalid function: {func}")
