from argparse import Namespace
from functools import partial
from typing import Callable

import torch

from buffer import ReplayBuffer
from energies import BaseEnergy
from gflownet_losses import get_gfn_loss
from langevin import langevin_dynamics
from models import GFN
from utils.misc_utils import get_exploration_std


def get_gfn_optimizer(
    gfn_model: GFN,
    lr_policy: float,
    lr_Z: float,
    lr_flow: float,
    lr_beta: float,
    lr_back: float,
    use_weight_decay=False,
    weight_decay=1e-7,
    use_scheduler=False,
    milestones: list[int] = [100000],
    gamma: float = 1.0,
):
    param_groups = [
        {"params": gfn_model.t_model.parameters()},
        {"params": gfn_model.s_model.parameters()},
        {"params": gfn_model.joint_model.parameters()},
    ]
    if gfn_model.lp_scaling_model is not None:
        param_groups += [{"params": gfn_model.lp_scaling_model.parameters()}]

    if isinstance(gfn_model.flow_model, torch.nn.Module):
        param_groups += [{"params": gfn_model.flow_model.parameters(), "lr": lr_flow}]
        if gfn_model.t_model_flow is not None:
            param_groups += [{"params": gfn_model.t_model_flow.parameters(), "lr": lr_flow}]
        if gfn_model.s_model_flow is not None:
            param_groups += [{"params": gfn_model.s_model_flow.parameters(), "lr": lr_flow}]
    else:
        param_groups += [{"params": [gfn_model.flow_model], "lr": lr_Z}]

    if gfn_model.beta_model is not None:
        param_groups += [{"params": gfn_model.beta_model, "lr": lr_beta}]

    if gfn_model.back_model is not None:
        param_groups += [{"params": gfn_model.back_model.parameters(), "lr": lr_back}]

    gfn_optimizer = torch.optim.Adam(
        param_groups, lr_policy, weight_decay=weight_decay if use_weight_decay else 0.0
    )

    gfn_scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(gfn_optimizer, milestones=milestones, gamma=gamma)
        if use_scheduler
        else None
    )
    return gfn_optimizer, gfn_scheduler


def train_step(
    energy: BaseEnergy,
    gfn_model: GFN,
    gfn_optimizer: torch.optim.Optimizer,
    gfn_scheduler: torch.optim.lr_scheduler.MultiStepLR | None,
    it: int,
    batch_size: int,
    loss_type: str,
    training_mode: str,
    bwd_from: str,
    discretizer: Callable[[int], torch.Tensor],
    exploratory: bool = False,
    exploration_factor: float = 0.0,
    exploration_wd: bool = False,
    buffer: ReplayBuffer | None = None,
    buffer_ls: ReplayBuffer | None = None,
    prefill: int = 0,
    local_search: bool = False,
    ls_args: Namespace | None = None,
    subtb_coef_matrix: torch.Tensor | None = None,
    clip_grad_norm: float = 1.0,
    device=torch.device("cpu"),
    resampling: bool = False,
    weighting: bool = False,
    aux_target: str = "target",  # target, loss, iw
    target_ess: float = 0.0,
    smoothing: str = "clip_above",
    alternating: bool = False,
):
    exploration_std = get_exploration_std(it, exploratory, exploration_factor, exploration_wd)

    run_forward = partial(
        fwd_train_step,
        energy=energy,
        gfn_model=gfn_model,
        batch_size=batch_size,
        loss_type=loss_type,
        discretizer=discretizer,
        subtb_coef_matrix=subtb_coef_matrix,
        exploration_std=exploration_std,
        buffer=buffer,
        device=device,
        aux_target=aux_target,
        target_ess=target_ess,
        smoothing=smoothing,
    )
    run_backward = partial(
        bwd_train_step,
        energy=energy,
        gfn_model=gfn_model,
        batch_size=batch_size,
        loss_type=loss_type,
        bwd_from=bwd_from,
        discretizer=discretizer,
        subtb_coef_matrix=subtb_coef_matrix,
        local_search=local_search,
        ls_args=ls_args,
        buffer=buffer,
        buffer_ls=buffer_ls,
        it=it,
        device=device,
    )

    if training_mode == "fwd":  # forward sampling only
        loss = run_forward(
            resampling=resampling & ((it % 2 == 1) if alternating else True),
            weighting=weighting & ((it % 2 == 1) if alternating else True),
        )

    elif training_mode == "bwd":  # backward sampling only
        assert bwd_from == "energy"  # FIXME: can buffer sampling be used here?
        loss = run_backward()

    elif training_mode == "both":  # Both forward and backward sampling
        # FIXME: is it worth to support different loss_type for fwd and bwd?
        assert (
            bwd_from == "buffer" and buffer is not None
        )  # FIXME: do we need to support bwd from energy?
        if it % 2 == 0 or it < prefill:
            loss = run_forward(
                resampling=resampling & ((it % 4 == 2) if alternating else True),
                weighting=weighting & ((it % 4 == 2) if alternating else True),
            )
        else:
            loss = run_backward()
    else:
        raise ValueError(f"Invalid training mode: {training_mode}")

    if it < prefill:
        return loss.item()

    loss.backward()
    if clip_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(gfn_model.parameters(), clip_grad_norm)
    gfn_optimizer.step()
    if gfn_scheduler is not None:
        gfn_scheduler.step()
    gfn_model.zero_grad()
    return loss.item()


def fwd_train_step(
    energy: BaseEnergy,
    gfn_model: GFN,
    batch_size: int,
    loss_type: str,
    discretizer: Callable[[int], torch.Tensor],
    subtb_coef_matrix: torch.Tensor | None,
    exploration_std=0.0,
    buffer: ReplayBuffer | None = None,
    device=torch.device("cpu"),
    resampling: bool = False,
    weighting: bool = False,
    aux_target: str = "target",  # target, loss, iw
    target_ess: float = 0.0,
    smoothing: str = "clip_above",
) -> torch.Tensor:
    if loss_type == "subtb":
        assert subtb_coef_matrix is not None

    init_states = torch.zeros(batch_size, energy.ndim).to(device)
    ts = discretizer(batch_size).to(device)

    states, log_pfs, log_pbs, log_fs, log_pfs_exp = gfn_model.get_trajectory_fwd(
        init_states, ts, exploration_std, energy.log_reward
    )
    with torch.no_grad():
        log_fs[:, -1] = energy.log_reward(states[:, -1])

    losses = get_gfn_loss(
        loss_type,
        log_pfs,
        log_pbs,
        log_fs,
        subtb_coef_matrix=subtb_coef_matrix,
        ndim=energy.ndim,
    )

    match aux_target:
        case "target":  # r(x)p_B(\tau|x)
            log_weights = (log_fs[:, -1] + log_pbs.sum(-1) - log_pfs_exp.sum(-1)).detach()
        case "loss":
            log_weights = (losses.log() - log_pfs_exp.sum(-1)).detach()
        case "iw":  # For TB, log_iw = log_r + log_pbs.sum(-1) - log_pfs.sum(-1) - log_Z
            log_weights = (log_fs[:, -1] + log_pbs.sum(-1) - 2 * log_pfs_exp.sum(-1)).detach()
        case _:
            raise ValueError(f"Invalid aux_target: {aux_target}")

    if target_ess != 0.0 and (
        (buffer is not None and buffer.prioritization == "normalized_iw")
        or (weighting or resampling)
    ):
        if 0.0 <= target_ess <= 1.0:
            target_ess = target_ess * batch_size
        else:
            assert 1.0 < target_ess <= batch_size, f"Invalid target ESS: {target_ess}"

        if ess(log_weights) >= target_ess:
            normalized_weights = log_weights.softmax(dim=0)
        else:
            if smoothing == "mix_with_uniform":
                normalized_weights = log_weights.softmax(dim=0)
                mixing_ratio = solve_mixing_ratio(normalized_weights, target_ess=target_ess)
                normalized_weights = (
                    1 - mixing_ratio
                ) * normalized_weights + mixing_ratio / batch_size
            else:
                func = get_func(smoothing)
                value, _ = binary_search(log_weights, target_ess, func)
                log_weights = func(log_weights, value)
                normalized_weights = log_weights.softmax(dim=0)
    else:
        normalized_weights = log_weights.softmax(dim=0)

    if buffer is not None:
        buffer.add(states[:, -1], log_fs[:, -1], losses.detach(), log_weights, normalized_weights)

    if weighting:
        loss = (normalized_weights * losses).sum()
    elif resampling:
        indices = torch.multinomial(normalized_weights, batch_size, replacement=True)
        loss = losses[indices].mean()
    else:
        loss = losses.mean()

    return loss


def bwd_train_step(
    energy: BaseEnergy,
    gfn_model: GFN,
    batch_size: int,
    loss_type: str,
    bwd_from: str,
    discretizer: Callable[[int], torch.Tensor],
    subtb_coef_matrix: torch.Tensor | None,
    local_search: bool = False,
    ls_args: Namespace | None = None,
    buffer: ReplayBuffer | None = None,
    buffer_ls: ReplayBuffer | None = None,
    it=0,
    device=torch.device("cpu"),
) -> torch.Tensor:
    if bwd_from == "energy":
        samples = energy.sample(batch_size).to(device)
        raise NotImplementedError("Training from energy is not used for this project.")

    elif bwd_from == "buffer":
        assert buffer is not None
        if local_search:
            assert buffer_ls is not None
            assert buffer_ls.prioritization in [
                "none",
                "reward",
            ], "Local search buffer cannot be prioritized by loss or iw"
            assert ls_args is not None
            if it % ls_args.ls_cycle < 2:
                samples, log_rs, _ = buffer.sample(batch_size)
                local_search_samples, log_rs = langevin_dynamics(
                    samples, energy.log_reward, device, ls_args
                )
                buffer_ls.add(local_search_samples, log_rs)
            samples, log_rs, indices = buffer_ls.sample(batch_size)
        else:
            samples, log_rs, indices = buffer.sample(batch_size)

        ts = discretizer(batch_size).to(device)
        _, log_pfs, log_pbs, log_fs = gfn_model.get_trajectory_bwd(samples, ts, energy.log_reward)

        log_fs[:, -1] = log_rs

        losses = get_gfn_loss(
            loss_type,
            log_pfs,
            log_pbs,
            log_fs,
            subtb_coef_matrix=subtb_coef_matrix,
            ndim=energy.ndim,
        )

        if buffer.prioritization == "loss":
            buffer.update(indices, samples, log_rs, losses.detach())

    return losses.mean()


###########################################
### Importance weight related functions ###
###########################################


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


def ess(log_weights: torch.Tensor) -> float:
    normalized_weights = log_weights.softmax(dim=0)
    return 1 / (normalized_weights**2).sum().item()


def binary_search(
    log_weights: torch.Tensor,
    target_ess: float,
    func: Callable[[torch.Tensor, float], torch.Tensor],
    tol=1e-2,
    max_steps=1000,
) -> tuple[float, int]:
    search_min, search_max = get_min_max(func, log_weights)

    steps = 0
    original_order = ess(func(log_weights, search_min)) < ess(func(log_weights, search_max))
    while True:
        steps += 1
        mid = (search_min + search_max) / 2
        if mid == search_min or mid == search_max:
            break  # Avoid meaningless loop; maybe the tolerance is too small

        log_weights_smoothed = func(log_weights, mid)
        _ess = ess(log_weights_smoothed)
        if (_ess > target_ess) == original_order:
            search_max = mid
        else:
            search_min = mid
        if abs(_ess - target_ess) < tol:
            break
        if steps > max_steps:
            raise ValueError(f"Binary search failed in {max_steps} steps")
    return mid, steps


def clip_below(log_weights: torch.Tensor, value: float) -> torch.Tensor:
    return log_weights.clamp(min=value)


def clip_above(log_weights: torch.Tensor, value: float) -> torch.Tensor:
    return log_weights.clamp(max=value)


def temper(log_weights: torch.Tensor, value: float) -> torch.Tensor:
    return log_weights / value


def get_func(smoothing: str) -> Callable[[torch.Tensor, float], torch.Tensor]:
    if smoothing == "clip_below":
        return clip_below
    elif smoothing == "clip_above":
        return clip_above
    elif smoothing == "temper":
        return temper
    else:
        raise ValueError(f"Invalid function: {smoothing}")


def get_min_max(func: Callable, log_weights: torch.Tensor) -> tuple[float, float]:
    if func == clip_above or func == clip_below:
        return log_weights.min().item(), log_weights.max().item()
    elif func == temper:
        return 1.0, 50.0
    else:
        raise ValueError(f"Invalid function: {func}")
