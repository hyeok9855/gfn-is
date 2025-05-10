import argparse
from functools import partial
import os

import torch
import wandb
from tqdm import trange

from buffers import TerminalStateBuffer, IntermediateStateBuffer
from discretizers import get_discretizer
from energies import IntermediateEnergy, get_energy
from losses import cal_subtb_coef_matrix
from models import GFN
from models.modules import get_module
from utils.eval_utils import eval_step
from utils.misc_utils import set_seed, get_name, linear_annealing
from utils.plot_utils import plot_step
from utils.train_utils import get_gfn_optimizer, train_step


def train(args):
    if "SLURM_PROCID" in os.environ:
        args.seed += int(os.environ["SLURM_PROCID"])
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    energy = get_energy(args, device)

    try:
        gt_xs = energy.sample(args.eval_data_size).to(device)
    except NotImplementedError:
        gt_xs = None

    energy_name = f"{args.energy_name}-{energy.ndim}d"
    exp_name = get_name(args)

    try:
        energy.cached_sample(args.eval_data_size)
    except NotImplementedError:
        pass

    # parent_dir = os.path.dirname(os.path.abspath(__file__))
    # save_dir = f"{parent_dir}/results/{energy_name}/{exp_name}"
    # os.makedirs(save_dir, exist_ok=True)

    config = args.__dict__
    config["Experiment"] = "{args.energy}"
    wandb.init(
        project=f"GFN-Diffusion-{energy_name}",
        config=config,
        name=exp_name,
        tags=[f"seed{args.seed}"],
        mode="disabled" if args.disable_wandb else "online",
    )

    module = get_module(args, energy)

    gfn_model = GFN(
        energy=energy,
        module=module,
        t_scale=args.t_scale,
        partial_energy=args.partial_energy,
        learn_beta_T=args.learn_beta_T,
        state_reduce_mean=args.state_reduce_mean,
        device=device,
    ).to(device)

    gfn_optimizer, gfn_scheduler = get_gfn_optimizer(
        gfn_model,
        lr_fwd=args.lr_fwd,
        lr_bwd=args.lr_bwd,
        lr_flow=args.lr_flow,
        lr_beta=args.lr_beta,
        use_weight_decay=args.use_weight_decay,
        weight_decay=args.weight_decay,
        use_scheduler=args.use_scheduler,
        milestones=[int(args.epochs * m) for m in args.milestones],
        gamma=args.gamma,
    )

    buffer = None
    if args.training_mode != "fwd" and args.bwd_from == "buffer":
        buffer_class = (
            TerminalStateBuffer if args.buffer_type == "terminal" else IntermediateStateBuffer
        )
        buffer = buffer_class(
            args.buffer_size,
            device,
            prioritization=args.prioritization,
            sampling_strategy=args.buffer_sampling,
            rank_k=args.rank_k,
            logr_lb=args.logr_lb,
        )

    subtb_coef_matrix = None
    if args.loss_type == "subtb" and args.subtb_n_chunks == 0:
        subtb_coef_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.T).to(device)

    train_discretizer = get_discretizer(
        discretizer=args.discretizer, max_ratio=args.discretizer_max_ratio
    )
    eval_discretizer = get_discretizer(discretizer="uniform")

    eval_step_partial = partial(
        eval_step,
        batch_size=args.batch_size,
        gfn_model=gfn_model,
        energy=energy,
        discretizer=eval_discretizer,
        T=args.eval_T,
        pis=args.loss_type == "pis",
        resampling=args.eval_resampling,
        resampling_strategy=args.resampling_strategy,
        weighting=args.eval_weighting,
        buffer=buffer if args.eval_buffer else None,
    )
    plot_step_partial = partial(plot_step, energy=energy)

    ######################
    # Main training loop #
    ######################

    gfn_model.train()
    for i in trange(args.epochs, dynamic_ncols=True):
        metrics = dict()

        ### Eval ###
        if i % args.eval_freq == 0:
            results, model_trajs, weights, sample_xs_r, buffer_xs = eval_step_partial(
                data_size=args.eval_data_size,
            )
            metrics.update(results)
            if i % args.plot_freq == 0:
                metrics.update(plot_step_partial(samples=model_trajs[:, -1]))
                if weights is not None:
                    metrics.update(
                        plot_step_partial(
                            samples=model_trajs[:, -1], weights=weights, suffix="_weighted"
                        )
                    )
                if sample_xs_r is not None:
                    metrics.update(plot_step_partial(samples=sample_xs_r, suffix="_resample"))
                if buffer_xs is not None:
                    metrics.update(plot_step_partial(samples=buffer_xs, suffix="_buffer"))

                # Plot intermediate states
                if len(args.plot_t_idx) > 0:
                    eval_ts = eval_discretizer(args.eval_data_size, args.eval_T)
                    for t_idx in args.plot_t_idx:
                        inter_states = model_trajs[:, t_idx]
                        assert (eval_ts[:, t_idx] == eval_ts[0, t_idx]).all()  # uniform discretizer
                        eval_t = round(eval_ts[0, t_idx].item(), 3)
                        inter_energy = IntermediateEnergy(energy, gfn_model, eval_t)
                        metrics.update(plot_step(inter_energy, inter_states, suffix=f"-t{eval_t}"))
                if buffer is not None and len(buffer) > 0 and len(args.plot_buffer_t_idx) > 0:
                    assert isinstance(buffer, IntermediateStateBuffer)
                    for t_idx in args.plot_buffer_t_idx:
                        inter_states, buf_ts, _ = buffer.sample_timestep(args.eval_data_size, t_idx)
                        assert (buf_ts == buf_ts[0]).all()  # uniform discretizer
                        buf_t = round(buf_ts[0].item(), 3)
                        inter_energy = IntermediateEnergy(energy, gfn_model, buf_t)
                        metrics.update(
                            plot_step(inter_energy, inter_states, suffix=f"_buffer-t{buf_t}")
                        )
            # if i % 1000 == 0:
            #     torch.save(gfn_model.state_dict(), f'{save_dir}/model.pt')

        ### Train ###

        epsilon = args.epsilon
        if args.anneal_epsilon:  # annealing half the epochs
            epsilon = linear_annealing(i, args.epochs // 2, 0.0, args.epsilon, descending=True)

        metrics["train/loss"] = train_step(
            energy,
            gfn_model,
            gfn_optimizer,
            gfn_scheduler,
            i,
            batch_size=args.batch_size,
            loss_type=args.loss_type,
            training_mode=args.training_mode,
            bwd_from=args.bwd_from,
            bwd_to_fwd_ratio=args.bwd_to_fwd_ratio,
            discretizer=train_discretizer,
            T=args.T,
            subtb_coef_matrix=subtb_coef_matrix,
            subtb_n_chunks=args.subtb_n_chunks,
            epsilon=epsilon,
            buffer=buffer,
            prefill=args.prefill,
            buffer_save_interval=args.buffer_save_interval,
            clip_grad_norm=args.clip_grad_norm,
            device=device,
            resampling=args.train_resampling,
            resampling_strategy=args.resampling_strategy,
            weighting=args.train_weighting,
            aux_target=args.aux_target,
            target_ess=args.target_ess,
            smoothing=args.smoothing,
            alternating=args.alternating,
        )
        wandb.log(metrics, step=i)

    ### Final eval ###
    try:
        energy.cached_sample(args.final_eval_data_size)
    except NotImplementedError:
        pass

    final_results, model_trajs, weights, sample_xs_r, buffer_xs = eval_step_partial(
        data_size=args.final_eval_data_size,
        final_eval=True,
    )
    metrics.update(final_results)
    metrics.update(plot_step_partial(samples=model_trajs[:, -1]))
    wandb.log(metrics, step=args.epochs)
    # torch.save(gfn_model.state_dict(), f'{save_dir}/model_final.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--energy_name",
        type=str,
        default="gmm40",
        choices=("25gmm", "gmm40", "funnel", "many_well", "lgcp", "lj13", "lj55"),
    )
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", default=False)

    parser.add_argument(
        "--loss_type",
        type=str,
        default="tb",
        choices=("tb", "db", "subtb", "logvar", "pis", "mle"),
    )
    parser.add_argument("--subtb_lambda", type=float, default=2.0)
    parser.add_argument("--subtb_n_chunks", type=int, default=0)

    parser.add_argument("--lr_fwd", type=float, default=1e-3)
    parser.add_argument("--lr_bwd", type=float, default=None)
    parser.add_argument("--lr_Z", type=float, default=1e-1)
    parser.add_argument("--lr_flow", type=float, default=1e-2)
    parser.add_argument("--lr_beta", type=float, default=1e-3)
    parser.add_argument("--use_weight_decay", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument("--use_scheduler", action="store_true", default=False)
    parser.add_argument("--milestones", type=float, nargs="+", default=[0.5, 0.9])
    parser.add_argument("--gamma", type=float, default=(10) ** (-1 / 2))

    parser.add_argument("--training_mode", type=str, default="both", choices=("fwd", "bwd", "both"))
    parser.add_argument("--bwd_from", type=str, default="buffer", choices=("energy", "buffer"))
    parser.add_argument("--bwd_to_fwd_ratio", type=int, default=1)
    parser.add_argument("--clip_grad_norm", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=25000)

    parser.add_argument("--module", type=str, default="pis_mlp", choices=("pis_mlp", "mlp", "egnn"))
    ################################################################
    ### MLP parameters
    parser.add_argument("--hidden_dim", type=int, default=256)
    # parser.add_argument("--s_emb_dim", type=int, default=256)
    # parser.add_argument("--t_emb_dim", type=int, default=256)
    # parser.add_argument("--harmonics_dim", type=int, default=256)
    parser.add_argument("--joint_layers", type=int, default=2)
    parser.add_argument("--no_zero_init", action="store_false", dest="zero_init")
    parser.add_argument("--no_share_embeddings", action="store_false", dest="share_embeddings")
    parser.add_argument("--flow_hidden_dim", type=int, default=256)
    # parser.add_argument("--flow_s_emb_dim", type=int, default=256)
    # parser.add_argument("--flow_t_emb_dim", type=int, default=256)
    # parser.add_argument("--flow_harmonics_dim", type=int, default=256)
    parser.add_argument("--flow_layers", type=int, default=2)
    parser.add_argument("--lp", action="store_true", default=False)
    parser.add_argument("--lp_scaling_per_dimension", action="store_true", default=False)
    parser.add_argument("--lgv_layers", type=int, default=3)
    parser.add_argument("--no_clipping", action="store_false", dest="clipping")
    parser.add_argument("--gfn_clip", type=float, default=1e4)
    parser.add_argument("--lgv_clip", type=float, default=1e2)
    parser.add_argument("--learn_pb", action="store_true", default=False)
    parser.add_argument("--pb_scale_range", type=float, default=0.1)
    parser.add_argument("--learn_variance", action="store_true", default=False)
    parser.add_argument("--log_var_range", type=float, default=4.0)

    parser.add_argument("--t_scale", type=float, default=1.0)
    parser.add_argument("--partial_energy", action="store_true", default=False)
    parser.add_argument("--learn_beta_T", type=int, default=0)
    ################################################################

    ################################################################
    ### For EGNN
    parser.add_argument("--egnn_hidden_nf", type=int, default=128)
    parser.add_argument("--egnn_n_layers", type=int, default=5)
    parser.add_argument("--egnn_no_recurrent", action="store_false", dest="egnn_recurrent")
    parser.add_argument("--egnn_no_attention", action="store_false", dest="egnn_attention")
    parser.add_argument(
        "--egnn_no_condition_time", action="store_false", dest="egnn_condition_time"
    )
    parser.add_argument("--egnn_no_tanh", action="store_false", dest="egnn_tanh")
    parser.add_argument("--egnn_agg", type=str, default="sum")
    ################################################################

    ################################################################
    ### For discretizer
    parser.add_argument("--T", type=int, default=100)
    # evaluation T
    parser.add_argument("--eval_T", type=int, default=100)
    # discretization scheme for training
    parser.add_argument(
        "--discretizer",
        type=str,
        default="uniform",
        choices=("uniform", "random", "equidistant"),
    )
    # maximum ratio between the longest and the shortest step size (only for 'random' discretizer)
    parser.add_argument("--discretizer_max_ratio", type=float, default=10.0)
    ################################################################

    ################################################################
    ### For replay buffer
    parser.add_argument("--buffer_size", type=int, default=-1)  # 100 * batch_size by default
    parser.add_argument(
        "--buffer_type", type=str, default="terminal", choices=("terminal", "intermediate")
    )
    # prioritization
    parser.add_argument(
        "--prioritization",
        type=str,
        default="none",
        choices=("none", "target", "loss", "normalized_iw"),
    )
    # buffer sampling strategy  # TODO: support percentile-based sampling
    parser.add_argument(
        "--buffer_sampling",
        type=str,
        default="multinomial",
        choices=("multinomial", "stratified", "systematic", "rank"),
    )
    # low rank_k give steep priorization in rank-based replay sampling
    parser.add_argument("--rank_k", type=float, default=1e-2)
    # logr_lb for filtering out samples with extremely low reward values for numerical stability
    parser.add_argument("--logr_lb", type=float, default=-1e5)
    # Interval between time indices at which to save intermediate states in the buffer
    # (0 means only save terminal states, n>0 saves states at every nth timestep)
    parser.add_argument("--buffer_save_interval", type=int, default=0)
    # prefill to wait before starting to sample from buffer
    parser.add_argument("--prefill", type=int, default=0)
    ################################################################

    ################################################################
    ### Exploration with extra noise
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--anneal_epsilon", action="store_true", default=False)
    ################################################################

    ################################################################
    ### Eval & Plot
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--eval_data_size", type=int, default=2000)
    parser.add_argument("--final_eval_data_size", type=int, default=2000)
    parser.add_argument("--plot_freq", type=int, default=2500)
    parser.add_argument("--plot_t_idx", type=int, nargs="+", default=[])
    parser.add_argument("--plot_buffer_t_idx", type=int, nargs="+", default=[])
    ################################################################

    ################################################################
    ### Importance sampling related
    parser.add_argument("--aux_target", type=str, default="target", choices=("target", "loss"))
    parser.add_argument("--train_resampling", action="store_true", default=False)
    parser.add_argument("--train_weighting", action="store_true", default=False)
    parser.add_argument("--alternating", action="store_true", default=False)
    parser.add_argument("--target_ess", type=float, default=0.0)  # 0.0 has no effect
    parser.add_argument(
        "--smoothing",
        type=str,
        default="temper",
        choices=("clip_above", "clip_below", "temper", "mix_with_uniform"),
    )
    parser.add_argument("--eval_resampling", action="store_true", default=False)
    parser.add_argument("--eval_weighting", action="store_true", default=False)
    parser.add_argument("--eval_buffer", action="store_true", default=False)
    parser.add_argument(
        "--resampling_strategy",
        type=str,
        default="multinomial",
        choices=("multinomial", "stratified", "systematic"),
    )
    ################################################################

    args = parser.parse_args()

    args.state_reduce_mean = True if args.energy_name in ["lj13", "lj55"] else False

    args.loss_type_str = args.loss_type
    if args.loss_type in ["db", "subtb"]:
        if args.partial_energy:
            args.loss_type_str = "fl-" + args.loss_type_str
        if args.loss_type == "subtb":
            if args.subtb_n_chunks > 0:
                args.loss_type_str += f"-nchunk{args.subtb_n_chunks}"
            else:
                args.loss_type_str += f"-lambda{args.subtb_lambda}"
        if args.learn_beta_T > 0:
            args.loss_type_str += f"-learnbetaT{args.learn_beta_T}"
    if args.learn_pb:
        args.loss_type_str += "-learnpb"

    if args.lr_bwd is None:
        args.lr_bwd = args.lr_fwd

    if args.loss_type == "pis":
        args.training_mode = "fwd"

    if args.loss_type in ["db", "subtb"]:
        args.conditional_flow_model = True
    else:
        args.conditional_flow_model = False
        args.lr_flow = args.lr_Z  # For TB

    if args.buffer_size == -1:
        args.buffer_size = 100 * args.batch_size

    assert args.plot_freq % args.eval_freq == 0

    train(args)
