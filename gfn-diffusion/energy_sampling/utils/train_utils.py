from argparse import Namespace
from functools import partial
from typing import cast

import torch

from buffer import ReplayBuffer
from energies import BaseEnergy
from langevin import langevin_dynamics
from models import GFN
from utils.misc_utils import get_exploration_std, get_gfn_forward_loss, get_gfn_backward_loss


def train_step(
    energy: BaseEnergy,
    gfn_model: GFN,
    gfn_optimizer: torch.optim.Optimizer,
    it: int,
    batch_size: int,
    training_loss: str,
    training_mode: str,
    bwd_from: str = "",
    exploratory: bool = False,
    exploration_factor: float = 0.0,
    exploration_wd: bool = False,
    buffer: ReplayBuffer | None = None,
    buffer_ls: ReplayBuffer | None = None,
    local_search: bool = False,
    ls_args: Namespace | None = None,
    subtb_coef_matrix: torch.Tensor | None = None,
    device=torch.device('cpu'),
):
    exploration_std = get_exploration_std(it, exploratory, exploration_factor, exploration_wd)

    run_forward = partial(
        fwd_train_step,
        energy=energy,
        gfn_model=gfn_model,
        batch_size=batch_size,
        training_loss=training_loss,
        subtb_coef_matrix=subtb_coef_matrix,
        exploration_std=exploration_std,
        device=device,
    )
    run_backward = partial(
        bwd_train_step,
        energy=energy,
        gfn_model=gfn_model,
        batch_size=batch_size,
        training_loss=training_loss,
        bwd_from=bwd_from,
        local_search=local_search,
        ls_args=ls_args,
        buffer=buffer,
        buffer_ls=buffer_ls,
        it=it,
        device=device,
    )

    if training_mode == "fwd":  # forward sampling only
        loss = cast(torch.Tensor, run_forward(return_exp=False))

    elif training_mode == "bwd":  # backward sampling only
        assert bwd_from == "energy"  # FIXME: can buffer sampling be used here?
        loss = run_backward()

    elif training_mode == "both":  # Both forward and backward sampling
        # FIXME: is it worth to support different training_loss for fwd and bwd?
        assert bwd_from == "buffer" and buffer is not None  # FIXME: do we need to support bwd from energy?
        if it % 2 == 0:
            loss, states, _, _, log_r  = run_forward(return_exp=True)
            buffer.add(states[:, -1], log_r)
        else:
            loss = run_backward()
    else:
        raise ValueError(f"Invalid training mode: {training_mode}")

    loss.backward()
    gfn_optimizer.step()
    gfn_model.zero_grad()
    return loss.item()


def fwd_train_step(
    energy: BaseEnergy,
    gfn_model: GFN,
    batch_size: int,
    training_loss: str,
    subtb_coef_matrix: torch.Tensor | None,
    exploration_std=0.0,
    return_exp=False,
    device=torch.device('cpu'),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
    if training_loss == 'subtb':
        assert subtb_coef_matrix is not None

    init_state = torch.zeros(batch_size, energy.ndim).to(device)
    return get_gfn_forward_loss(
        training_loss,
        init_state,
        gfn_model,
        energy.log_reward,
        subtb_coef_matrix,
        exploration_std=exploration_std,
        return_exp=return_exp,
    )


def bwd_train_step(
    energy: BaseEnergy,
    gfn_model: GFN,
    batch_size: int,
    training_loss: str,
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
        assert buffer is not None and buffer_ls is not None
        if local_search:
            assert ls_args is not None
            if it % ls_args.ls_cycle < 2:
                samples, _ = buffer.sample()
                local_search_samples, log_r = langevin_dynamics(samples, energy.log_reward, device, ls_args)
                buffer_ls.add(local_search_samples, log_r)
            samples, _ = buffer_ls.sample()
        else:
            samples, _ = buffer.sample()

    loss = get_gfn_backward_loss(training_loss, samples, gfn_model, energy.log_reward)
    return loss
