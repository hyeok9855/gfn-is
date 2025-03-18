from argparse import Namespace
from functools import partial
from typing import cast

import torch

from buffer import ReplayBuffer
from energies import BaseEnergy
from gflownet_losses import bwd_mle, bwd_tb, bwd_tb_avg, db, fwd_tb, fwd_tb_avg, subtb
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
    clip_grad_norm: float = 1.0,
    bwd_from: str = "",
    exploratory: bool = False,
    exploration_factor: float = 0.0,
    exploration_wd: bool = False,
    buffer: ReplayBuffer | None = None,
    buffer_ls: ReplayBuffer | None = None,
    warmup_steps: int = 0,
    local_search: bool = False,
    ls_args: Namespace | None = None,
    subtb_coef_matrix: torch.Tensor | None = None,
    device=torch.device('cpu'),
    resampling: bool = False,
    weighting: bool = False,
    aux_reward: str = "reward",  # reward, loss, delta
    temperature: float = 1.0,
):
    exploration_std = get_exploration_std(it, exploratory, exploration_factor, exploration_wd)

    run_forward = partial(
        fwd_train_step,
        energy=energy,
        gfn_model=gfn_model,
        batch_size=batch_size,
        loss_type=loss_type,
        subtb_coef_matrix=subtb_coef_matrix,
        exploration_std=exploration_std,
        buffer=buffer,
        device=device,
        resampling=resampling,
        weighting=weighting,
        aux_reward=aux_reward,
        temperature=temperature
    )
    run_backward = partial(
        bwd_train_step,
        energy=energy,
        gfn_model=gfn_model,
        batch_size=batch_size,
        loss_type=loss_type,
        bwd_from=bwd_from,
        local_search=local_search,
        ls_args=ls_args,
        buffer=buffer,
        buffer_ls=buffer_ls,
        it=it,
        device=device,
    )

    if training_mode == "fwd":  # forward sampling only
        loss = cast(torch.Tensor, run_forward())

    elif training_mode == "bwd":  # backward sampling only
        assert bwd_from == "energy"  # FIXME: can buffer sampling be used here?
        loss = run_backward()

    elif training_mode == "both":  # Both forward and backward sampling
        # FIXME: is it worth to support different loss_type for fwd and bwd?
        assert bwd_from == "buffer" and buffer is not None  # FIXME: do we need to support bwd from energy?
        if it % 2 == 0 or it < warmup_steps:
            loss = run_forward()
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
    subtb_coef_matrix: torch.Tensor | None,
    exploration_std=0.0,
    buffer: ReplayBuffer | None = None,
    device=torch.device('cpu'),
    resampling: bool = False,
    weighting: bool = False,
    aux_reward: str = "reward",  # reward, loss, delta
    temperature: float = 1.0,
) -> torch.Tensor:
    if loss_type == 'subtb':
        assert subtb_coef_matrix is not None

    init_state = torch.zeros(batch_size, energy.ndim).to(device)
    delta, loss, states, log_pfs, log_pbs, log_r = get_gfn_forward_loss(
        loss_type,
        init_state,
        gfn_model,
        energy.log_reward,
        subtb_coef_matrix,
        exploration_std=exploration_std,
    )
    if buffer is not None:
        buffer.add(states[:, -1], log_r, delta)

    if resampling or weighting:
        match aux_reward:
            case "reward":
                log_weights = log_r + log_pbs.sum(-1) - log_pfs.sum(-1)
            case "loss":
                log_weights = loss.log() + log_pbs.sum(-1) - log_pfs.sum(-1)
            case "delta":
                assert delta is not None  # For TB, delta = log_r + log_pbs.sum(-1) - log_pfs.sum(-1) - log_Z
                log_weights = (1e-2 + delta.clamp(min=0)).log() + log_pbs.sum(-1) - log_pfs.sum(-1)
            case _:
                raise ValueError(f"Invalid aux_reward: {aux_reward}")
        log_weights = log_weights.detach()

        normalized_weights = (log_weights / temperature).softmax(dim=0)

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
    local_search: bool = False,
    ls_args: Namespace | None = None,
    buffer: ReplayBuffer | None = None,
    buffer_ls: ReplayBuffer | None = None,
    it=0,
    device=torch.device('cpu'),
) -> torch.Tensor:
    if bwd_from == 'energy':
        samples = energy.sample(batch_size).to(device)

    elif bwd_from == 'buffer':
        assert buffer is not None
        if local_search:
            assert buffer_ls is not None
            assert buffer_ls.prioritization in ["none", "reward"], "Local search buffer cannot be prioritized by loss or delta"
            assert ls_args is not None
            if it % ls_args.ls_cycle < 2:
                samples, log_r, _, _ = buffer.sample(batch_size)
                local_search_samples, log_r = langevin_dynamics(samples, energy.log_reward, device, ls_args)
                buffer_ls.add(local_search_samples, log_r)
            samples, log_r, _, _ = buffer_ls.sample(batch_size)
            delta, loss = get_gfn_backward_loss(loss_type, samples, gfn_model, energy.log_reward)
        else:
            samples, log_r, _, indices = buffer.sample(batch_size)
            delta, loss = get_gfn_backward_loss(loss_type, samples, gfn_model, energy.log_reward)
            buffer.update(indices, samples, log_r, delta)

    return loss.mean()


def get_gfn_forward_loss(
    loss_type,
    init_state,
    gfn_model,
    log_reward,
    coeff_matrix,
    exploration_std=0.0,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if loss_type == 'tb':
        delta, states, log_pfs, log_pbs, log_r = fwd_tb(init_state, gfn_model, log_reward, exploration_std)
        loss = 0.5 * (delta ** 2)
    elif loss_type == 'tb-avg':
        delta, states, log_pfs, log_pbs, log_r = fwd_tb_avg(init_state, gfn_model, log_reward, exploration_std)
        loss = 0.5 * (delta ** 2)
    elif loss_type == 'db':
        delta = None
        loss, states, log_pfs, log_pbs, log_r = db(init_state, gfn_model, log_reward, exploration_std)
    elif loss_type == 'subtb':
        delta = None
        loss, states, log_pfs, log_pbs, log_r = subtb(init_state, gfn_model, log_reward, coeff_matrix, exploration_std)
    else:
        raise ValueError(f'Invalid training loss for forward: {loss_type}')

    return delta, loss, states, log_pfs, log_pbs, log_r


def get_gfn_backward_loss(
    loss_type,
    samples,
    gfn_model,
    log_reward,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if loss_type == 'tb':
        delta = bwd_tb(samples, gfn_model, log_reward)
        loss = 0.5 * (delta ** 2)
    elif loss_type == 'tb-avg':
        delta = bwd_tb_avg(samples, gfn_model, log_reward)
        loss = 0.5 * (delta ** 2)
    elif loss_type == 'mle':
        delta = None
        loss = bwd_mle(samples, gfn_model, log_reward)
    else:
        raise ValueError(f'Invalid training loss for backward: {loss_type}')
    return delta, loss
