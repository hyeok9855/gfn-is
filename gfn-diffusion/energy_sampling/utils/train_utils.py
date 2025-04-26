from functools import partial
from typing import Callable, Literal

import torch

from buffers import BaseBuffer, TerminalStateBuffer, IntermediateStateBuffer
from energies import BaseEnergy
from gflownet_losses import get_gfn_loss
from models import GFN
from utils.misc_utils import get_exploration_std
from utils.sampling_utils import get_sampling_func


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
    discretizer: Callable[[int, int], torch.Tensor],
    T: int,
    exploratory: bool = False,
    exploration_factor: float = 0.0,
    exploration_wd: bool = False,
    buffer: BaseBuffer | None = None,
    prefill: int = 0,
    subtb_coef_matrix: torch.Tensor | None = None,
    subtb_chunk_size: int = 0,
    clip_grad_norm: float = 1.0,
    device=torch.device("cpu"),
    weighting: bool = False,
    resampling: bool = False,
    resampling_strategy: Literal["multinomial", "stratified", "systematic"] = "multinomial",
    aux_target: str = "target",  # target, loss
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
        T=T,
        subtb_coef_matrix=subtb_coef_matrix,
        subtb_chunk_size=subtb_chunk_size,
        exploration_std=exploration_std,
        buffer=buffer,
        device=device,
        resampling_strategy=resampling_strategy,
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
        T=T,
        subtb_coef_matrix=subtb_coef_matrix,
        subtb_chunk_size=subtb_chunk_size,
        buffer=buffer,
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
    discretizer: Callable[[int, int], torch.Tensor],
    T: int,
    subtb_coef_matrix: torch.Tensor | None,
    subtb_chunk_size: int = 0,
    exploration_std=0.0,
    buffer: BaseBuffer | None = None,
    device=torch.device("cpu"),
    weighting: bool = False,
    resampling: bool = False,
    resampling_strategy: Literal["multinomial", "stratified", "systematic"] = "multinomial",
    aux_target: str = "target",  # target, loss
    target_ess: float = 0.0,
    smoothing: str = "clip_above",
) -> torch.Tensor:
    init_states = torch.zeros(batch_size, energy.ndim).to(device)
    ts = discretizer(batch_size, T).to(device)

    # Forward sampling
    states, log_pfs, log_pbs, log_fs, log_pfs_exp = gfn_model.get_trajectory_fwd(
        init_states, ts, energy.log_reward, exploration_std=exploration_std
    )

    # Compute losses
    losses = get_gfn_loss(
        loss_type,
        log_pfs,
        log_pbs,
        log_fs,
        subtb_coef_matrix=subtb_coef_matrix,
        subtb_chunk_size=subtb_chunk_size,
        ndim=energy.ndim,
    )

    normalized_iws_0t = None
    if (buffer is not None and buffer.prioritization == "normalized_iw") or weighting or resampling:
        # Compute importance weights
        log_iws_0t = torch.zeros(batch_size, ts.shape[1] - 1).to(device)  # shape: (bs, T)
        aux_target_measure_0t = torch.zeros_like(log_iws_0t)
        match aux_target:
            case "target":  # r(x)p_B(\tau|x)
                aux_target_measure_0t = log_fs[:, 1:] + log_pbs.cumsum(-1)
            case "loss":
                assert loss_type in ["tb", "tb-avg", "pis"]
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

    # Add data to buffer
    if buffer is not None:
        data_dict = {"states": states[:, 1:], "log_fs": log_fs[:, 1:]}  # (bs, T) both
        if buffer.prioritization == "loss":
            assert isinstance(buffer, TerminalStateBuffer)
            data_dict["losses"] = losses.unsqueeze(1)  # (bs, 1)
        elif buffer.prioritization == "normalized_iw":
            assert normalized_iws_0t is not None
            data_dict["normalized_iws"] = normalized_iws_0t  # (bs, T)

        if isinstance(buffer, TerminalStateBuffer):
            data_dict = {k: v[:, -1] for k, v in data_dict.items()}
        else:  # IntermediateStateBuffer
            data_dict["ts"] = ts[:, 1:]  # (bs, T)
            if subtb_chunk_size > 0:
                data_dict = {
                    k: v[:, subtb_chunk_size - 1 :: subtb_chunk_size] for k, v in data_dict.items()
                }
        buffer.add(**data_dict)

    # Weighting or resampling here is supported only on the trajectory level
    # TODO: support for the transition-level weighting or resampling
    if weighting or resampling:
        assert normalized_iws_0t is not None
        x_iws = normalized_iws_0t[:, -1]
        if weighting:
            loss = (x_iws * losses).sum()
        else:  # resampling
            indices = get_sampling_func(resampling_strategy)(x_iws, batch_size, True)
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
    discretizer: Callable[[int, int], torch.Tensor],
    T: int,
    subtb_coef_matrix: torch.Tensor | None,
    subtb_chunk_size: int = 0,
    buffer: BaseBuffer | None = None,
    device=torch.device("cpu"),
) -> torch.Tensor:
    if bwd_from == "energy":
        raise NotImplementedError("Training from energy is not used for this project.")
        samples = energy.sample(batch_size).to(device)

    elif bwd_from == "buffer":
        assert buffer is not None
        if isinstance(buffer, TerminalStateBuffer):
            buf_xs, buf_log_rs, indices = buffer.sample(batch_size)
            # each with shape (bs,)

            # Construct complete trajectories
            ts = discretizer(batch_size, T).to(device)
            _, log_pfs, log_pbs, log_fs = gfn_model.get_trajectory_bwd(
                buf_xs, ts, buf_log_rs, energy.log_reward
            )

        elif isinstance(buffer, IntermediateStateBuffer):
            # # Option 1: Construct transitions / subtrajectories (chunks)
            # n_chunks = T // subtb_chunk_size  # assumption: T is divisible by subtb_chunk_size
            # buf_states, buf_ts, buf_log_fs, indices = buffer.sample(batch_size * n_chunks)
            # # each with shape (bs,)

            # # TODO: support for other discretizers
            # assert discretizer.__name__ == "uniform_discretizer"
            # sub_ts = discretizer(batch_size * n_chunks, subtb_chunk_size).to(device) / n_chunks

            # # clamp to avoid negative time steps from floating point error
            # ts = (buf_ts.unsqueeze(-1) + sub_ts - sub_ts[:, [-1]]).clamp(min=0.0)
            # _, log_pfs, log_pbs, log_fs = gfn_model.get_subtrajectory_bwd(
            #     buf_states, ts, buf_log_fs, energy.log_reward
            # )

            # Option 2: Construct complete trajectories by sampling both backward and forward
            buf_states, buf_ts, _, indices = buffer.sample(batch_size)

            # TODO: support for other discretizers
            assert discretizer.__name__ == "uniform_discretizer"
            ts = discretizer(batch_size, T).to(device)
            _, log_pfs, log_pbs, log_fs, _ = gfn_model.get_trajectory_fwd_and_bwd(
                buf_states, ts, buf_ts, energy.log_reward, exploration_std=0.0
            )

        else:
            raise ValueError(f"Invalid buffer type: {type(buffer)}")

        losses = get_gfn_loss(
            loss_type,
            log_pfs,
            log_pbs,
            log_fs,
            subtb_coef_matrix=subtb_coef_matrix,
            subtb_chunk_size=subtb_chunk_size,
            ndim=energy.ndim,
        )

        if buffer.prioritization == "loss":
            buffer.update(indices, losses=losses)

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
        if (mid == search_min).all() or (mid == search_max).all():
            break  # Avoid meaningless loop; maybe the tolerance is too small

        new_log_weights = func(log_weights, mid)  # shape: (bs, T)
        new_ess = ess(log_weights=new_log_weights)  # shape: (T,)
        new_dones = (~dones) & (abs(new_ess - target_ess) < tol)  # shape: (T,)
        log_weights_smoothed[:, new_dones] = new_log_weights[:, new_dones]
        dones = dones | new_dones

        search_max = torch.where((new_ess > target_ess) == original_order, mid, search_max)
        search_min = torch.where((new_ess < target_ess) == original_order, mid, search_min)

        if steps > max_steps:
            raise ValueError(f"Binary search failed in {max_steps} steps")
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
    if func == clip_above or func == clip_below:
        return log_weights.min().item(), log_weights.max().item()
    elif func == temper:
        return 1.0, 100.0
    else:
        raise ValueError(f"Invalid function: {func}")
