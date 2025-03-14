import random
import math

import numpy as np
import PIL
import torch

from gflownet_losses import (
    bwd_mle, bwd_tb, bwd_tb_avg, db, fwd_tb, fwd_tb_avg, subtb
)
from models import GFN


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def cal_subtb_coef_matrix(lamda: float, N: int) -> torch.Tensor:
    """
    diff_matrix: (N+1, N+1)
    0, 1, 2, ...
    -1, 0, 1, ...
    -2, -1, 0, ...

    self.coef[i, j] = lamda^(j-i) / total_lambda  if i < j else 0.
    """
    range_vals = torch.arange(N + 1)
    diff_matrix = range_vals - range_vals.view(-1, 1)
    B = np.log(lamda) * diff_matrix
    B[diff_matrix <= 0] = -np.inf
    log_total_lambda = torch.logsumexp(B.view(-1), dim=0)
    coef = torch.exp(B - log_total_lambda)
    return coef


def get_gfn_optimizer(
    gfn_model: GFN,
    lr_policy,
    lr_flow,
    lr_back,
    back_model=False,
    conditional_flow_model=False,
    use_weight_decay=False,
    weight_decay=1e-7,
):
    param_groups = [
        {'params': gfn_model.t_model.parameters()},
        {'params': gfn_model.s_model.parameters()},
        {'params': gfn_model.joint_model.parameters()},
    ]
    if gfn_model.lp_scaling_model is not None:
        param_groups += [{'params': gfn_model.lp_scaling_model.parameters()}]

    if conditional_flow_model:
        assert isinstance(gfn_model.flow_model, torch.nn.Module)
        param_groups += [{'params': gfn_model.flow_model.parameters(), 'lr': lr_flow}]
    else:
        param_groups += [{'params': [gfn_model.flow_model], 'lr': lr_flow} ]

    if back_model:
        assert gfn_model.back_model is not None
        param_groups += [{'params': gfn_model.back_model.parameters(), 'lr': lr_back}]

    if use_weight_decay:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy, weight_decay=weight_decay)
    else:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy)

    return gfn_optimizer


def get_gfn_forward_loss(
    training_loss,
    init_state,
    gfn_model,
    log_reward,
    coeff_matrix,
    exploration_std=0.0,
    return_exp=False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
    if training_loss == 'tb':
        output = fwd_tb(init_state, gfn_model, log_reward, exploration_std, return_exp=return_exp)
    elif training_loss == 'tb-avg':
        output = fwd_tb_avg(init_state, gfn_model, log_reward, exploration_std, return_exp=return_exp)
    elif training_loss == 'db':
        output = db(init_state, gfn_model, log_reward, exploration_std)
    elif training_loss == 'subtb':
        output = subtb(init_state, gfn_model, log_reward, coeff_matrix, exploration_std)
    else:
        raise ValueError(f'Invalid training loss for forward: {training_loss}')

    if return_exp:
        loss, states, log_pfs, log_pbs, log_r = output
        return loss, states, log_pfs, log_pbs, log_r
    else:
        loss = output
        return loss


def get_gfn_backward_loss(
    training_loss,
    samples,
    gfn_model,
    log_reward,
) -> torch.Tensor:
    if training_loss == 'tb':
        loss = bwd_tb(samples, gfn_model, log_reward)
    elif training_loss == 'tb-avg':
        loss = bwd_tb_avg(samples, gfn_model, log_reward)
    elif training_loss == 'mle':
        loss = bwd_mle(samples, gfn_model, log_reward)
    else:
        raise ValueError(f'Invalid training loss for backward: {training_loss}')
    return loss


def get_exploration_std(
    iter,
    exploratory,
    exploration_factor=0.1,
    exploration_wd=False,
) -> float:
    if exploratory is False:
        return 0.0
    if exploration_wd:
        exploration_std = exploration_factor * max(0, 1. - iter / 5000.)
    else:
        exploration_std = exploration_factor
    return exploration_std


def get_name(args):
    name = ''
    if args.lp:
        name = f'lp_'
        if args.lp_scaling_per_dimension:
            name = f'lp_scaling_per_dimension_'
    if args.exploratory and (args.exploration_factor is not None):
        if args.exploration_wd:
            name = f'exploration_wd_{args.exploration_factor}_{name}_'
        else:
            name = f'exploration_{args.exploration_factor}_{name}_'

    if args.learn_pb:
        name = f'{name}learn_pb_scale_range_{args.pb_scale_range}_'

    if args.clipping:
        name = f'{name}clipping_lgv_{args.lgv_clip}_gfn_{args.gfn_clip}_'

    if args.training_loss == 'subtb':
        training_loss = f'subtb_lambda_{args.subtb_lambda}'
        if args.partial_energy:
            training_loss = f'{training_loss}_{args.partial_energy}'
    else:
        training_loss = args.training_loss

    ways = args.training_mode
    if args.local_search:
        local_search = f'local_search_iter_{args.max_iter_ls}_burn_{args.burn_in}_cycle_{args.ls_cycle}_step_{args.ld_step}_beta_{args.beta}_rankw_{args.rank_weight}_prioritized_{args.prioritized}'
        ways = f'{ways}/{local_search}'

    if args.pis_architectures:
        results = 'results_pis_architectures'
    else:
        results = 'results'

    name = f'{results}/{args.target_energy}/{name}gfn/{ways}/T_{args.T}/tscale_{args.t_scale}/lvr_{args.log_var_range}/'

    name = f'{name}/seed_{args.seed}/'

    return name
