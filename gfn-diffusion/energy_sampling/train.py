import argparse
import os

import torch
import wandb
from tqdm import trange

from buffers import (
    PIWIntermediateStateBuffer,
    PIWTerminalStateBuffer,
    IntermediateStateBuffer,
    TerminalStateBuffer,
)
from discretizers import get_discretizer
from energies import get_energy
from mcmcs import MALA, MD
from models import GFN
from models.modules import get_module
from trainer import Trainer
from utils.misc_utils import get_name, set_seed
from utils.sampling_utils import get_sampling_func
from utils.train_utils import get_gfn_optimizer


def train(args):
    if "SLURM_PROCID" in os.environ:
        args.seed += int(os.environ["SLURM_PROCID"])
    set_seed(args.seed)

    if args.precision == "double":
        torch.set_default_dtype(torch.float64)

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

    buffer = mcmc = None
    if args.use_buffer:
        if args.prioritization == "iw":
            buffer_class = (
                PIWTerminalStateBuffer
                if args.buffer_type == "terminal"
                else PIWIntermediateStateBuffer
            )
        else:
            buffer_class = (
                TerminalStateBuffer if args.buffer_type == "terminal" else IntermediateStateBuffer
            )

        buffer = buffer_class(
            args.buffer_size,
            device,
            prioritization=args.prioritization,
            sampling_func=get_sampling_func(args.buffer_sampling, args.rank_k),
            logr_lb=args.logr_lb,
            smoothing_strategy=args.smoothing_strategy,
            target_ess=args.target_ess,
        )

        if args.mcmc_type != "none":
            mcmc_args = {
                "energy": energy,
                "n_steps": args.mcmc_n_steps,
                "burn_in": args.mcmc_burn_in,
                "thinning": args.mcmc_thinning,
                "step_size": args.mcmc_step_size,
                "gamma": args.mcmc_gamma,  # For MD
            }
            if args.mcmc_type == "md":
                mcmc = MD(**mcmc_args)
            elif args.mcmc_type == "mala":
                mcmc = MALA(**mcmc_args)
            else:
                raise ValueError(f"Invalid MCMC type: {args.mcmc_type}")

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
        bwd_to_fwd_ratio=args.bwd_to_fwd_ratio,
        buffer=buffer,
        buffer_save_interval=args.buffer_save_interval,
        prefill_epochs=args.prefill_epochs,
        batch_size=args.batch_size,
        train_discretizer=train_discretizer,
        train_T=args.T,
        epsilon=args.epsilon,
        anneal_epsilon=args.anneal_epsilon,
        weighting=args.train_weighting,
        resampling=args.train_resampling,
        resampling_strategy=args.resampling_strategy,
        alternating=args.alternating,
        target_ess=args.target_ess,
        smoothing_strategy=args.smoothing_strategy,
        mcmc=mcmc,
        mcmc_freq=args.mcmc_freq,
        mcmc_batch_size=args.mcmc_batch_size,
        invtemp=args.invtemp,
        invtemp_anneal=args.invtemp_anneal,
        init_log_Z=args.init_log_Z,
        eval_batch_size=args.eval_batch_size,
        eval_discretizer=eval_discretizer,
        eval_T=args.eval_T,
        eval_weighting=args.eval_weighting,
        eval_resampling=args.eval_resampling,
        plot_gt=args.plot_gt,
        plot_t_idx=args.plot_t_idx,
        plot_buffer_t_idx=args.plot_buffer_t_idx,
    )

    ######################
    # Main training loop #
    ######################

    pbar = trange(args.epochs, desc="[Train]", dynamic_ncols=True)
    eubo_cache = elbo_cache = ess_cache = float("nan")
    for it in pbar:
        metrics = dict()

        ### Eval and plot###
        if it % args.eval_freq == 0:
            metrics.update(
                trainer.eval_and_plot(
                    data_size=args.eval_data_size,
                    full_eval=True if (it % args.full_eval_freq == 0 and args.full_eval) else False,
                    plot=args.plot if it % args.plot_freq == 0 else False,
                )
            )
            eubo_cache = metrics["eval/eubo"]
            elbo_cache = metrics["eval/elbo"]
            ess_cache = metrics["eval/ess(%)"]

        ### Train ###
        metrics["train/loss"] = trainer.train_step(it)
        pbar.set_postfix(
            {
                "Loss": metrics["train/loss"],
                "EUBO": eubo_cache,
                "ELBO": elbo_cache,
                "ESS": ess_cache,
            }
        )

        ### Log ###
        wandb.log(metrics, step=it)

    ### Final eval and plot ###
    final_metrics = trainer.eval_and_plot(
        data_size=args.final_eval_data_size,
        full_eval=True if args.full_eval else False,
        final_eval=True,
        plot=args.plot,
    )
    wandb.log(final_metrics, step=args.epochs)
    desc = ""
    if final_metrics.get("final_eval/eubo-elbo") is not None:
        desc += f"{'EUBO-ELBO':<10}: {final_metrics['final_eval/eubo-elbo']:.3f}\n"
    else:
        desc += f"{'ELBO':<10}: {final_metrics['final_eval/elbo']:.3f}\n"
    if final_metrics.get("final_eval/Sinkhorn") is not None:
        desc += f"{'Sinkhorn':<10}: {final_metrics['final_eval/Sinkhorn']:.3f}\n"
    if final_metrics.get("final_eval/ess(%)") is not None:
        desc += f"{'ESS':<10}: {final_metrics['final_eval/ess(%)']:.3f}\n"
    print(f"===============\n[Final results]\n{desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--energy_name",
        type=str,
        default="gmm40",
        choices=(
            "25gmm",
            "gmm40",
            "student_t_mixture",
            "manywell",
            "funnel",
            "lgcp",
            "lj13",
            "lj55",
            "aldp",
            "aldp_fab",
        ),
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
    parser.add_argument("--lr_Z", type=float, default=5e-2)
    parser.add_argument("--lr_flow", type=float, default=1e-2)
    parser.add_argument("--lr_beta", type=float, default=1e-3)
    parser.add_argument("--lr_lgv", type=float, default=1e-4)
    parser.add_argument("--use_weight_decay", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument("--use_scheduler", action="store_true", default=False)
    parser.add_argument("--milestones", type=float, nargs="+", default=[0.5, 0.9])
    parser.add_argument("--gamma", type=float, default=0.3)

    parser.add_argument("--bwd_to_fwd_ratio", type=float, default=1.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--eval_batch_size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=25000)

    parser.add_argument("--module", type=str, default="pismlp", choices=("pismlp", "mlp", "egnn"))
    parser.add_argument("--use_checkpoint", action="store_true", default=False)
    parser.add_argument("--init_log_Z", type=str, default="0.0")  # "iw_elbo", "elbo" or float
    parser.add_argument("--precision", type=str, default="float", choices=("float", "double"))

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
    parser.add_argument("--no_use_buffer", action="store_false", dest="use_buffer")
    parser.add_argument("--buffer_size", type=int, default=-1)  # 100 * batch_size by default
    parser.add_argument(
        "--buffer_type", type=str, default="terminal", choices=("terminal", "intermediate")
    )
    # prioritization
    parser.add_argument(
        "--prioritization",
        type=str,
        default="none",
        choices=("none", "target", "loss", "iw", "normalized_iw"),
    )
    # buffer sampling strategy  # TODO: support percentile-based sampling
    parser.add_argument(
        "--buffer_sampling",
        type=str,
        default="systematic",
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
    parser.add_argument("--prefill_epochs", type=int, default=-1)
    ################################################################

    ################################################################
    ### For MCMC
    parser.add_argument("--mcmc_type", type=str, default="none", choices=("none", "md", "mala"))
    parser.add_argument("--mcmc_freq", type=int, default=100)
    parser.add_argument("--mcmc_batch_size", type=int, default=100)
    parser.add_argument("--mcmc_n_steps", type=int, default=1000)
    parser.add_argument("--mcmc_burn_in", type=int, default=100)
    parser.add_argument("--mcmc_thinning", type=int, default=1)
    parser.add_argument("--mcmc_step_size", type=float, default=0.001)
    parser.add_argument("--mcmc_gamma", type=float, default=1.0)  # for MD
    ################################################################

    ### Exploration with extra noise
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--no_anneal_epsilon", action="store_false", dest="anneal_epsilon")
    ################################################################

    ################################################################
    ### Inverse temperature of the energy
    parser.add_argument("--invtemp", type=float, default=1.0)
    parser.add_argument("--no_invtemp_anneal", action="store_false", dest="invtemp_anneal")
    ################################################################

    ################################################################
    ### Eval & Plot
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--eval_data_size", type=int, default=2000)
    parser.add_argument("--final_eval_data_size", type=int, default=2000)
    parser.add_argument("--no_full_eval", action="store_false", dest="full_eval")
    parser.add_argument("--full_eval_freq", type=float, default=0.1)
    parser.add_argument("--no_plot", action="store_false", dest="plot")
    parser.add_argument("--plot_freq", type=float, default=0.1)
    parser.add_argument("--no_plot_gt", action="store_false", dest="plot_gt")
    parser.add_argument("--plot_t_idx", type=int, nargs="+", default=[])
    parser.add_argument("--plot_buffer_t_idx", type=int, nargs="+", default=[])
    ################################################################

    ################################################################
    ### Importance sampling related
    parser.add_argument("--train_resampling", action="store_true", default=False)
    parser.add_argument("--train_weighting", action="store_true", default=False)
    parser.add_argument("--alternating", action="store_true", default=False)
    parser.add_argument("--target_ess", type=float, default=0.0)  # 0.0 has no effect
    parser.add_argument(
        "--smoothing_strategy",
        type=str,
        default="temper",
        choices=("clip_above", "clip_below", "temper", "mix_with_uniform"),
    )
    parser.add_argument("--eval_resampling", action="store_true", default=False)
    parser.add_argument("--eval_weighting", action="store_true", default=False)
    parser.add_argument(
        "--resampling_strategy",
        type=str,
        default="systematic",
        choices=("multinomial", "stratified", "systematic"),
    )
    ################################################################

    args = parser.parse_args()

    try:
        args.init_log_Z = float(args.init_log_Z)
    except ValueError:
        assert args.init_log_Z in ["iw_elbo", "elbo"]

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

    if args.loss_type == "mle" or args.loss_type == "pis":
        args.use_buffer = False

    if args.loss_type in ["db", "subtb"]:
        args.conditional_flow_model = True
    else:
        args.conditional_flow_model = False
        args.lr_flow = args.lr_Z  # For TB

    if args.buffer_size == -1:
        args.buffer_size = 100 * args.batch_size

    if args.prefill_epochs == -1:
        args.prefill_epochs = min(100, args.buffer_size / args.batch_size // 10)

    if args.full_eval_freq < 1:
        args.full_eval_freq = args.full_eval_freq * args.epochs
    if args.plot_freq < 1:
        args.plot_freq = args.plot_freq * args.epochs
    args.full_eval_freq = int(args.full_eval_freq)
    args.plot_freq = int(args.plot_freq)

    assert args.plot_freq % args.eval_freq == 0

    train(args)
