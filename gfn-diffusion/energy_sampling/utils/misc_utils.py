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


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def get_exploration_std(
    iter,
    exploratory,
    exploration_factor=0.1,
    exploration_wd=False,
) -> float:
    if exploratory is False:
        return 0.0
    if exploration_wd:
        exploration_std = exploration_factor * max(0, 1.0 - iter / 5000.0)
    else:
        exploration_std = exploration_factor
    return exploration_std


def get_name(args):
    name = ""

    name += args.loss_type
    if args.loss_type == "subtb":
        name += f"-lambda{args.subtb_lambda}"
    if args.lp:
        name += "-lp"
    if args.partial_energy:
        name += "-partialE"

    name += f"_t_scale{args.t_scale}-NNdim{args.hidden_dim}"
    if args.loss_type in ["subtb", "db"]:
        name += f"-Fdim{args.flow_hidden_dim}"

    name += f"-lr{args.lr_policy}"
    if args.loss_type == "tb":
        name += f"-lrZ{args.lr_Z}"
    elif args.loss_type in ["subtb", "db"]:
        name += f"-lrflow{args.lr_flow}"
    if args.use_weight_decay:
        name += f"-wd{args.weight_decay}"
    if args.use_scheduler:
        name += f"-lrsch"

    name += f"_{args.training_mode}"
    if args.training_mode != "fwd":
        name += f"-{args.bwd_from}"
        if args.bwd_from == "buffer":
            buffer_size_str = (
                f"{args.buffer_size // 1000}K"
                if args.buffer_size >= 1000
                else f"{args.buffer_size}"
            )
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

    name += f"_{args.exp_name}" if args.exp_name else ""

    return name


def torch_quantile(  # Ref: https://github.com/pytorch/pytorch/issues/64947#issuecomment-2304371451
    input: torch.Tensor,
    q: float | torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "higher",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Sanitization: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Sanitization: inteporlation
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = math.floor
    elif interpolation == "higher":
        inter = math.ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Sanitization: out
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Logic
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Rectification: keepdim
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)


def huber_loss(value: torch.Tensor, quantile: float = 1.0) -> torch.Tensor:
    """
    Huber loss with a given quantile.
    """
    if quantile == 0.0:
        return value.abs()
    elif quantile == 1.0:
        return value**2
    else:
        value_abs = value.abs()
        cutoff = torch_quantile(value_abs, quantile)
        return value_abs * value_abs.clamp(max=cutoff)
