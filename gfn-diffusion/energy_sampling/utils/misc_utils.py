import contextlib
import random
import math

import numpy as np
import torch

from models import GFN


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


@contextlib.contextmanager
def temp_seed(seed):
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    torch_cuda_states = torch.cuda.get_rng_state_all()
    set_seed(seed)

    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(torch_cuda_states)


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def cal_subtb_coef_matrix(lamda: float, N: int) -> torch.Tensor:
    """
    diff_matrix: (N+1, N+1)
     0,  1,  2, ...,   N
    -1,  0,  1, ..., N-1
    -2, -1,  0, .... N-2
    ...

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
    lr_policy: float,
    lr_Z: float,
    lr_flow: float,
    lr_back: float,
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
        param_groups += [{'params': [gfn_model.flow_model], 'lr': lr_Z} ]

    if back_model:
        assert gfn_model.back_model is not None
        param_groups += [{'params': gfn_model.back_model.parameters(), 'lr': lr_back}]

    gfn_optimizer = torch.optim.Adam(param_groups, lr_policy, weight_decay=weight_decay if use_weight_decay else 0.0)
    return gfn_optimizer


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

    name += args.loss_type
    if args.loss_type == "subtb":
        name += f"-lambda{args.subtb_lambda}"
    if args.lp:
        name += "-lp"
    if args.partial_energy:
        name += "-partialE"

    name += f"_t_scale{args.t_scale}-NNhidden{args.hidden_dim}"

    name += f"-lr{args.lr_policy}-lrflow{args.lr_flow}"
    if args.use_weight_decay:
        name += f"-wd{args.weight_decay}"

    name += f"_{args.training_mode}"
    if args.training_mode != "fwd":
        name += f"-{args.bwd_from}"
        if args.bwd_from == "buffer":
            buffer_size_str = f"{args.buffer_size // 1000}K" if args.buffer_size >= 1000 else f"{args.buffer_size}"
            name += f"-{buffer_size_str}"
            if args.prioritization != "none":
                name += f"-{args.prioritization}-{args.buffer_sampling}"

    name += f"-T{args.T}-{args.discretizer}"
    if args.discretizer == "random":
        name += f"-maxr{args.max_ratio}"

    if args.exploratory:
        name += f"_expl{args.exploration_factor}"
        if args.exploration_wd:
            name += "wd"

    if args.train_resampling or args.train_weighting:
        if args.train_resampling:
            name += "_resampling"
        if args.train_weighting:
            name += "_weighting"
        name += f"-{args.aux_target}"
        if args.alternating:
            name += "-alt"

    if args.target_ess != 0.0:
        name += f"_tgtess{args.target_ess}-{args.smoothing}"

    name += f"_sd{args.seed}"
    name += f"_{args.exp_name}" if args.exp_name else ""

    return name
