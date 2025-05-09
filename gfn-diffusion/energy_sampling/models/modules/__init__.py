import argparse

from energies import BaseEnergy

from .base import BaseModule
from .egnn_modules import EGNNModule
from .mlp_modules import MLPModule
from .pis_mlp_modules import PISMLPModule


def get_module(args: argparse.Namespace, energy: BaseEnergy) -> BaseModule:
    if "mlp" in args.module:
        mlp_kwargs = {
            "ndim": energy.ndim,
            "harmonics_dim": args.hidden_dim,
            "t_emb_dim": args.hidden_dim,
            "s_emb_dim": args.hidden_dim,
            "hidden_dim": args.hidden_dim,
            "joint_layers": args.joint_layers,
            "zero_init": args.zero_init,
            "conditional_flow_model": args.conditional_flow_model,
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
        }

        module_cls = MLPModule if args.module == "mlp" else PISMLPModule
        return module_cls(**mlp_kwargs)

    elif "egnn" in args.module:
        n_particles = getattr(energy, "n_particles")
        spatial_dim = getattr(energy, "spatial_dim")

        egnn_kwargs = {
            "n_particles": n_particles,
            "spatial_dim": spatial_dim,
            "hidden_nf": args.egnn_hidden_nf,
            "n_layers": args.egnn_n_layers,
            "conditional_flow_model": args.conditional_flow_model,
            "recurrent": args.egnn_recurrent,
            "attention": args.egnn_attention,
            "condition_time": args.egnn_condition_time,
            "tanh": args.egnn_tanh,
            "agg": args.egnn_agg,
        }
        return EGNNModule(**egnn_kwargs)
    else:
        raise ValueError(f"Module {args.module} not found")
