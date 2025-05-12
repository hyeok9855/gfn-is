import argparse
import os

import torch
import wandb
from tqdm import trange

from buffers import IntermediateStateBuffer, TerminalStateBuffer
from discretizers import get_discretizer
from energies import get_energy
from models import GFN
from models.modules import get_module
from trainer import Trainer
from utils.misc_utils import get_name, set_seed
from utils.train_utils import get_gfn_optimizer


def train(args):
    if "SLURM_PROCID" in os.environ:
        args.seed += int(os.environ["SLURM_PROCID"])
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    energy = get_energy(args, device)
    exp_name = get_name(args)

    config = args.__dict__
    wandb.init(
        project=f"GFN-Diffusion-{args.energy_name}-{energy.ndim}d",
        config=config,
        name=exp_name,
        tags=[f"seed{args.seed}"],
        mode="disabled" if args.disable_wandb else "online",
    )

    #########################
    # Initialize components #
    #########################

    module = get_module(args, energy)

    gfn_model = GFN(
        energy=energy,
        module=module,
        t_scale=args.t_scale,
        partial_energy=args.partial_energy,
        learn_beta_T=args.learn_beta_T,
        state_remove_mean=args.state_remove_mean,
        device=device,
    ).to(device)

    gfn_optimizer, gfn_scheduler = get_gfn_optimizer(
        gfn_model,
        lr_fwd=args.lr_fwd,
        lr_bwd=args.lr_bwd,
        lr_flow=args.lr_flow,
        lr_beta=args.lr_beta,
        lr_lgv=args.lr_lgv,
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

    train_discretizer = get_discretizer(
        discretizer=args.discretizer, max_ratio=args.discretizer_max_ratio
    )
    eval_discretizer = get_discretizer(discretizer="uniform")

    trainer = Trainer(
        energy=energy,
        gfn_model=gfn_model,
        optimizer=gfn_optimizer,
        scheduler=gfn_scheduler,
        clip_grad_norm=args.clip_grad_norm,
        loss_type=args.loss_type,
        subtb_lambda=args.subtb_lambda,
        subtb_n_chunks=args.subtb_n_chunks,
        sublogvar_K=args.sublogvar_K,
        n_epochs=args.epochs,
        training_mode=args.training_mode,
        bwd_from=args.bwd_from,
        bwd_to_fwd_ratio=args.bwd_to_fwd_ratio,
        buffer=buffer,
        buffer_save_interval=args.buffer_save_interval,
        prefill_epochs=args.prefill,
        batch_size=args.batch_size,
        train_discretizer=train_discretizer,
        train_T=args.T,
        epsilon=args.epsilon,
        anneal_epsilon=args.anneal_epsilon,
        weighting=args.train_weighting,
        resampling=args.train_resampling,
        resampling_strategy=args.resampling_strategy,
        alternating=args.alternating,
        aux_target=args.aux_target,
        target_ess=args.target_ess,
        smoothing_strategy=args.smoothing,
        eval_discretizer=eval_discretizer,
        eval_T=args.eval_T,
        eval_weighting=args.eval_weighting,
        eval_resampling=args.eval_resampling,
        plot_t_idx=args.plot_t_idx,
        plot_buffer_t_idx=args.plot_buffer_t_idx,
    )

    ######################
    # Main training loop #
    ######################

    pbar = trange(args.epochs, dynamic_ncols=True)
    for it in pbar:
        metrics = dict()

        ### Eval and plot###
        if it % args.eval_freq == 0:
            metrics.update(
                trainer.eval_and_plot(
                    data_size=args.eval_data_size,
                    full_eval=True if it % args.full_eval_freq == 0 else False,
                    plot=args.plot if it % args.plot_freq == 0 else False,
                )
            )
            if metrics.get("eval/eubo-elbo") is not None:
                desc = f"EUBO-ELBO: {metrics['eval/eubo-elbo']:.3f}"
            else:
                desc = f"ELBO: {metrics['eval/elbo']:.3f}"
            pbar.set_description(desc)

        ### Train ###
        metrics["train/loss"] = trainer.train_step(it)
        pbar.set_postfix({"Loss": metrics["train/loss"]})

        ### Log ###
        wandb.log(metrics, step=it)

    ### Final eval and plot ###
    final_metrics = trainer.eval_and_plot(
        data_size=args.final_eval_data_size, full_eval=True, final_eval=True, plot=args.plot
    )
    wandb.log(final_metrics, step=args.epochs)
    desc = ""
    if final_metrics.get("eval/eubo-elbo") is not None:
        desc += f"{'EUBO-ELBO':<10}: {final_metrics['eval/eubo-elbo']:.3f}\n"
    else:
        desc += f"{'ELBO':<10}: {final_metrics['eval/elbo']:.3f}\n"
    if final_metrics.get("eval/Sinkhorn") is not None:
        desc += f"{'Sinkhorn':<10}: {final_metrics['eval/Sinkhorn']:.3f}\n"
    if final_metrics.get("eval/ess") is not None:
        desc += f"{'ESS':<10}: {final_metrics['eval/ess']:.3f}\n"
    print(f"Final results:\n {desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--energy_name",
        type=str,
        default="gmm40",
        choices=("25gmm", "gmm40", "funnel", "manywell", "lgcp", "lj13", "lj55"),
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
    parser.add_argument("--sublogvar_K", type=int, default=1)

    parser.add_argument("--lr_fwd", type=float, default=1e-3)
    parser.add_argument("--lr_bwd", type=float, default=None)
    parser.add_argument("--lr_Z", type=float, default=1e-1)
    parser.add_argument("--lr_flow", type=float, default=1e-2)
    parser.add_argument("--lr_beta", type=float, default=1e-3)
    parser.add_argument("--lr_lgv", type=float, default=1e-4)
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

    parser.add_argument("--module", type=str, default="pismlp", choices=("pismlp", "mlp", "egnn"))
    parser.add_argument("--use_checkpoint", action="store_true", default=False)
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
    parser.add_argument("--out_clip", type=float, default=1e4)
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
    parser.add_argument("--full_eval_freq", type=int, default=2500)
    parser.add_argument("--eval_data_size", type=int, default=2000)
    parser.add_argument("--final_eval_data_size", type=int, default=2000)
    parser.add_argument("--no_plot", action="store_false", dest="plot")
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

    args.state_remove_mean = True if args.energy_name in ["lj13", "lj55"] else False

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
