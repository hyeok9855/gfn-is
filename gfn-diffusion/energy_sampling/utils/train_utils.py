from argparse import Namespace
from functools import partial
from typing import Callable

import torch

from buffer import ReplayBuffer
from energies import BaseEnergy
from gflownet_losses import db_loss, subtb_loss, tb_avg_loss, tb_loss
from langevin import langevin_dynamics
from models import GFN
from utils.misc_utils import get_exploration_std


def train_step(
    energy: BaseEnergy,
    gfn_model: GFN,
    gfn_optimizer: torch.optim.Optimizer,
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
    warmup_steps: int = 0,
    local_search: bool = False,
    ls_args: Namespace | None = None,
    subtb_coef_matrix: torch.Tensor | None = None,
    clip_grad_norm: float = 1.0,
    device=torch.device('cpu'),
    resampling: bool = False,
    weighting: bool = False,
    aux_reward: str = "reward",  # reward, loss, log_iw
    target_ess: float = 1.0,
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
        aux_reward=aux_reward,
        target_ess=target_ess,
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
        assert bwd_from == "buffer" and buffer is not None  # FIXME: do we need to support bwd from energy?
        if it % 2 == 0 or it < warmup_steps:
            loss = run_forward(
                resampling=resampling & ((it % 4 == 2) if alternating else True),
                weighting=weighting & ((it % 4 == 2) if alternating else True),
            )
        else:
            loss = run_backward()
    else:
        raise ValueError(f"Invalid training mode: {training_mode}")

    loss.backward()
    if clip_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(gfn_model.parameters(), clip_grad_norm)
    gfn_optimizer.step()
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
    device=torch.device('cpu'),
    resampling: bool = False,
    weighting: bool = False,
    aux_reward: str = "reward",  # reward, loss, log_iw
    target_ess: float = 0.0,
) -> torch.Tensor:
    if loss_type == 'subtb':
        assert subtb_coef_matrix is not None

    init_states = torch.zeros(batch_size, energy.ndim).to(device)
    ts = discretizer(batch_size).to(device)

    states, log_pfs, log_pbs, log_fs, log_pfs_exp = gfn_model.get_trajectory_fwd(
        init_states, ts, exploration_std, energy.log_reward
    )
    with torch.no_grad():
        log_fs[:, -1] = energy.log_reward(states[:, -1])

    log_iw, loss = get_gfn_loss(
        loss_type,
        log_pfs,
        log_pbs,
        log_fs,
        subtb_coef_matrix=subtb_coef_matrix,
        ndim=energy.ndim,
    )

    if buffer is not None:
        buffer.add(states[:, -1], log_fs[:, -1], log_iw)

    if resampling or weighting:
        if target_ess != 0.0:
            if 0.0 <= target_ess <= 1.0:
                target_ess = target_ess * batch_size
            else:
                assert 1.0 < target_ess <= batch_size, f"Invalid target ESS: {target_ess}"
            assert target_ess > 1.0

        if target_ess == batch_size:  # No need to weighting or resampling
            return loss.mean()

        log_pfs_exp = log_pfs_exp if exploration_std > 0.0 else log_pfs
        match aux_reward:
            case "reward":  # r(x)p_B(\tau|x)
                log_weights = log_fs[:, -1] + log_pbs.sum(-1) - log_pfs_exp.sum(-1)
            case "loss":
                log_weights = loss.log() - log_pfs_exp.sum(-1)
            case "iw":
                assert log_iw is not None  # For TB, log_iw = log_r + log_pbs.sum(-1) - log_pfs.sum(-1) - log_Z
                log_weights = log_iw - log_pfs_exp.sum(-1)
            case _:
                raise ValueError(f"Invalid aux_reward: {aux_reward}")
        log_weights = log_weights.detach()

        normalized_weights = log_weights.softmax(dim=0)
        if target_ess != 0.0:
            mixing_ratio = solve_mixing_ratio(normalized_weights, target_ess=target_ess)
            # Mix the weights with uniform distribution
            normalized_weights = (1 - mixing_ratio) * normalized_weights + mixing_ratio / batch_size

        if weighting:
            return (normalized_weights * loss).sum()
        else:  # resampling
            indices = torch.multinomial(normalized_weights, batch_size, replacement=True)
            return loss[indices].mean()
    else:
        return loss.mean()


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
    device=torch.device('cpu'),
) -> torch.Tensor:
    if bwd_from == 'energy':
        samples = energy.sample(batch_size).to(device)
        raise NotImplementedError("Training from energy is not used for this project.")

    elif bwd_from == 'buffer':
        assert buffer is not None
        if local_search:
            assert buffer_ls is not None
            assert buffer_ls.prioritization in ["none", "reward"], (
                "Local search buffer cannot be prioritized by loss or log_iw"
            )
            assert ls_args is not None
            if it % ls_args.ls_cycle < 2:
                samples, log_r, _, _ = buffer.sample(batch_size)
                local_search_samples, log_r = langevin_dynamics(samples, energy.log_reward, device, ls_args)
                buffer_ls.add(local_search_samples, log_r)
            samples, log_r, _, indices = buffer_ls.sample(batch_size)
        else:
            samples, log_r, _, indices = buffer.sample(batch_size)

        ts = discretizer(batch_size).to(device)
        _, log_pfs, log_pbs, log_fs = gfn_model.get_trajectory_bwd(
            samples, ts, energy.log_reward
        )

        log_fs[:, -1] = log_r

        log_iw, loss = get_gfn_loss(
            loss_type,
            log_pfs,
            log_pbs,
            log_fs,
            subtb_coef_matrix=subtb_coef_matrix,
            ndim=energy.ndim,
        )

        buffer.update(indices, samples, log_r, log_iw)

    return loss.mean()


def get_gfn_loss(
    loss_type: str,
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    subtb_coef_matrix: torch.Tensor | None = None,
    ndim: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if loss_type == 'tb':
        log_iw, loss = tb_loss(log_pfs, log_pbs, log_fs[:, 0], log_fs[:, -1])
    elif loss_type == 'tb-avg':
        log_iw, loss = tb_avg_loss(log_pfs, log_pbs, log_fs[:, -1])
    elif loss_type == 'db':
        log_iw, loss = db_loss(log_pfs, log_pbs, log_fs)
    elif loss_type == 'subtb':
        assert subtb_coef_matrix is not None
        log_iw, loss = subtb_loss(log_pfs, log_pbs, log_fs, subtb_coef_matrix)
    elif loss_type == 'pis':
        assert ndim is not None
        log_iw = (log_fs[:, -1] + log_pbs.sum(-1) - log_pfs.sum(-1)).detach()
        loss = (1 / ndim) * (log_pfs.sum(-1) - log_pbs.sum(-1) - log_fs[:, -1])
    else:
        raise ValueError(f'Invalid training loss: {loss_type}')

    return log_iw, loss


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
    nw_squared_sum = (normalized_weights ** 2).sum().item()

    ess_before = 1 / nw_squared_sum
    if ess_before >= target_ess:
        return 0.0

    A = nw_squared_sum - 2 * nw_sum / N + 1 / N
    B = nw_squared_sum - nw_sum / N
    C = nw_squared_sum - 1 / target_ess

    min_lhs = C - B ** 2 / A
    if min_lhs >= 0:
        raise ValueError(f"Cannot achieve target ESS: {target_ess}")
        return 1.0

    mixing_ratio_1 = (B + (B ** 2 - A * C) ** 0.5) / A
    mixing_ratio_2 = (B - (B ** 2 - A * C) ** 0.5) / A

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
