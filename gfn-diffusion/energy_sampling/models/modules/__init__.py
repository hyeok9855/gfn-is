import argparse

import torch

from .base import BaseModule
from .mlp_modules import MLPModule
from .pis_mlp_modules import PISMLPModule


def get_module(args: argparse.Namespace, device: torch.device) -> BaseModule:
    mlp_kwargs = {
        "ndim": args.ndim,
        "conditional_flow_model": args.conditional_flow_model,
        "harmonics_dim": args.hidden_dim,
        "t_emb_dim": args.hidden_dim,
        "s_emb_dim": args.hidden_dim,
        "hidden_dim": args.hidden_dim,
        "joint_layers": args.joint_layers,
        "zero_init": args.zero_init,
        "share_embeddings": args.share_embeddings,
        "flow_harmonics_dim": args.flow_hidden_dim,
        "flow_t_emb_dim": args.flow_hidden_dim,
        "flow_s_emb_dim": args.flow_hidden_dim,
        "flow_hidden_dim": args.flow_hidden_dim,
        "flow_layers": args.flow_layers,
        "lp": args.lp,
        "lp_scaling_per_dimension": args.lp_scaling_per_dimension,
        "lgv_layers": args.lgv_layers,
        "clipping": args.clipping,
        "lgv_clip": args.lgv_clip,
        "gfn_clip": args.gfn_clip,
        "learn_pb": args.learn_pb,
        "pb_scale_range": args.pb_scale_range,
        "learn_variance": args.learn_variance,
        "log_var_range": args.log_var_range,
        "device": device,
    }

    if args.module == "mlp":
        return MLPModule(**mlp_kwargs)
    elif args.module == "pis_mlp":
        return PISMLPModule(**mlp_kwargs)
    else:
        raise ValueError(f"Module {args.module} not found")
