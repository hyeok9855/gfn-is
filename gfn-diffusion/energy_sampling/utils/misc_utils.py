import argparse
import contextlib
import math
import random

import numpy as np
import torch


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


def logmeanexp(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return x.logsumexp(dim) - math.log(x.shape[dim])


def linear_annealing(
    current: int,
    n_rounds: int,
    min_val: float,
    max_val: float,
    descending=False,
    log=False,
    avoid_zero=False,
) -> float:
    assert min_val <= max_val
    if min_val == max_val:
        return min_val

    start_val, end_val = min_val, max_val
    if descending:
        start_val, end_val = end_val, start_val

    if current >= n_rounds:
        return end_val

    num = current + 1 if avoid_zero else current
    denom = n_rounds + 1 if avoid_zero else n_rounds
    multiplier = math.log(num) / math.log(denom) if log else num / denom
    return start_val + (end_val - start_val) * multiplier


def get_name(args: argparse.Namespace) -> str:
    name = ""

    name += args.loss_type_str

    name += f"_{args.module}"
    if args.module in ["mlp", "pismlp", "ddsmlp"]:
        name += f"-h{args.hidden_dim}l{args.joint_layers}"
        if args.loss_type in ["subtb", "db"]:
            name += f"-Fh{args.flow_hidden_dim}l{args.flow_layers}"
    elif args.module == "egnn":
        name += f"-h{args.egnn_hidden_nf}l{args.egnn_n_layers}"

    if args.lp:
        name += "-lp"

    if args.partial_energy:
        name += "_partialE"
        if args.learn_beta_T > 0:
            name += f"-learnbetaT{args.learn_beta_T}"

    name += f"_tscale{args.t_scale}"
    name += f"_bsz{args.batch_size}"

    name += f"_lrfwd{args.lr_fwd}"
    if args.loss_type == "tb":
        name += f"-lrZ{args.lr_Z}"
    elif args.loss_type in ["subtb", "db"]:
        name += f"-lrflow{args.lr_flow}"
        if args.learn_beta_T > 0:
            name += f"-lrbeta{args.lr_beta}"
    if args.learn_pb:
        name += f"-lrbwd{args.lr_bwd}"
    if args.use_weight_decay:
        name += f"-wd{args.weight_decay}"
    if args.use_scheduler:
        name += f"-lrsch"

    name += f"_T{args.T}-{args.discretizer}"
    if args.discretizer == "random":
        name += f"-maxr{args.discretizer_max_ratio}"

    if args.epsilon > 0.0:
        name += f"_eps{args.epsilon}"
        if args.anneal_epsilon:
            name += f"-annealed"

    if args.invtemp != 1.0:
        name += f"_invtemp{args.invtemp}"
        if args.invtemp_anneal:
            name += "-annealed"

    if args.use_buffer:
        name += f"_btf{args.bwd_to_fwd_ratio}"
        name += f"-buf{args.buffer_type}"
        buffer_size_str = (
            f"{args.buffer_size // 1000}K" if args.buffer_size >= 1000 else f"{args.buffer_size}"
        )
        name += f"-{buffer_size_str}"
        if args.prioritization != "none":
            name += f"-{args.prioritization}-{args.buffer_sampling}"

        if args.mcmc_type != "none":
            name += f"_{args.mcmc_type}-freq{args.mcmc_freq}"
            name += f"-n{args.mcmc_n_steps}-b{args.mcmc_burn_in}-s{args.mcmc_step_size}"
            if args.mcmc_type == "md":
                name += f"-gamma{args.mcmc_gamma}"

    if args.train_resampling or args.train_weighting:
        if args.train_resampling:
            name += "_resampling"
        if args.train_weighting:
            name += "_weighting"
        if args.alternating:
            name += "-alt"

    if args.target_ess != 0.0:
        name += f"_tgtess{args.target_ess}-{args.smoothing_strategy}"

    name += f"_{args.exp_name}" if args.exp_name else ""

    return name
