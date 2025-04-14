import argparse
from functools import partial
import os
from copy import deepcopy

import torch
import wandb
from tqdm import trange

from buffer import ReplayBuffer
from discretizers import get_discretizer
from energies import get_energy
from gflownet_losses import cal_subtb_coef_matrix
from models import GFN
from utils.eval_utils import eval_step
from utils.misc_utils import get_name, set_seed
from utils.plot_utils import plot_step
from utils.train_utils import get_gfn_optimizer, train_step


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    energy = get_energy(args.energy_name, args.ndim, device=device)

    energy_name = f"{args.energy_name}-{args.ndim}d"
    exp_name = get_name(args)

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
    )

    subtb_coef_matrix = None
    if args.loss_type == "subtb":
        subtb_coef_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.T).to(device)

    ls_args = None
    if args.local_search:
        ls_args = argparse.Namespace(
            max_iter_ls=args.max_iter_ls,
            burn_in=args.burn_in,
            ls_cycle=args.ls_cycle,
            ld_step=args.ld_step,
            ld_schedule=args.ld_schedule,
            target_acceptance_rate=args.target_acceptance_rate,
        )

    try:
        gt_xs = energy.sample(args.eval_data_size).to(device)
    except NotImplementedError:
        gt_xs = None

    gfn_model = GFN(
        energy.ndim,
        args.hidden_dim,
        args.hidden_dim,
        args.hidden_dim,
        args.hidden_dim,
        flow_harmonics_dim=args.flow_hidden_dim,
        flow_t_emb_dim=args.flow_hidden_dim,
        flow_s_emb_dim=args.flow_hidden_dim,
        flow_hidden_dim=args.flow_hidden_dim,
        log_var_range=args.log_var_range,
        t_scale=args.t_scale,
        lp=args.lp,
        learned_variance=args.learned_variance,
        partial_energy=args.partial_energy,
        clipping=args.clipping,
        lgv_clip=args.lgv_clip,
        gfn_clip=args.gfn_clip,
        pb_scale_range=args.pb_scale_range,
        lp_scaling_per_dimension=args.lp_scaling_per_dimension,
        conditional_flow_model=args.conditional_flow_model,
        share_embeddings=args.share_embeddings,
        learn_pb=args.learn_pb,
        pis_architectures=args.pis_architectures,
        lgv_layers=args.lgv_layers,
        joint_layers=args.joint_layers,
        zero_init=args.zero_init,
        device=device,
    ).to(device)

    gfn_optimizer, gfn_scheduler = get_gfn_optimizer(
        gfn_model,
        args.lr_policy,
        args.lr_Z,
        args.lr_flow,
        args.lr_back,
        args.learn_pb,
        args.conditional_flow_model,
        args.use_weight_decay,
        args.weight_decay,
        args.use_scheduler,
        [int(args.epochs * m) for m in args.milestones],
        args.gamma,
    )

    buffer = ReplayBuffer(
        args.buffer_size,
        device,
        prioritization=args.prioritization,
        sampling_strategy=args.buffer_sampling,
        rank_k=args.rank_k,
        logr_lb=args.logr_lb,
    )
    buffer_ls = None
    if args.local_search:
        buffer_ls = deepcopy(buffer)
        buffer_ls.prioritization = "reward"

    train_discretizer = get_discretizer(
        discretizer=args.discretizer, T=args.T, max_ratio=args.discretizer_max_ratio
    )
    eval_discretizer = get_discretizer(discretizer="uniform", T=args.eval_T)

    eval_step_partial = partial(
        eval_step,
        gt_xs=gt_xs,
        gfn_model=gfn_model,
        energy=energy,
        discretizer=eval_discretizer,
        pis=args.loss_type == "pis",
        resampling=args.eval_resampling,
        weighting=args.eval_weighting,
        buffer=buffer if args.eval_buffer else None,
    )
    plot_step_partial = partial(
        plot_step,
        energy=energy,
        resampling=args.eval_resampling,
        weighting=args.eval_weighting,
    )

    ######################
    # Main training loop #
    ######################

    gfn_model.train()
    for i in trange(args.epochs, dynamic_ncols=True):
        metrics = dict()

        ### Eval ###
        if i % args.eval_freq == 0:
            results, model_trajs, weights, model_trajs_r = eval_step_partial(args.eval_data_size)
            metrics.update(results)
            if i % args.plot_freq == 0:
                images = plot_step_partial(
                    samples=model_trajs[:, -1], resampled_samples=model_trajs_r, weights=weights
                )
                metrics.update(images)
            # if i % 1000 == 0:
            #     torch.save(gfn_model.state_dict(), f'{save_dir}/model.pt')

        ### Train ###
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
            discretizer=train_discretizer,
            exploratory=args.exploratory,
            exploration_factor=args.exploration_factor,
            exploration_wd=args.exploration_wd,
            buffer=buffer,
            buffer_ls=buffer_ls,
            prefill=args.prefill,
            local_search=args.local_search,
            ls_args=ls_args,
            subtb_coef_matrix=subtb_coef_matrix,
            clip_grad_norm=args.clip_grad_norm,
            device=device,
            resampling=args.train_resampling,
            weighting=args.train_weighting,
            aux_target=args.aux_target,
            target_ess=args.target_ess,
            smoothing=args.smoothing,
            alternating=args.alternating,
        )
        wandb.log(metrics, step=i)

    ### Final eval ###
    final_results, model_trajs, weights, model_trajs_r = eval_step_partial(
        args.final_eval_data_size, final_eval=True
    )
    metrics.update(final_results)
    final_images = plot_step_partial(
        samples=model_trajs[:, -1], resampled_samples=model_trajs_r, weights=weights
    )
    metrics.update(final_images)
    wandb.log(metrics, step=args.epochs)
    # torch.save(gfn_model.state_dict(), f'{save_dir}/model_final.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--energy_name",
        type=str,
        default="gmm40",
        choices=("25gmm", "gmm40", "funnel", "many_well", "lgcp"),
    )
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", default=False)

    parser.add_argument(
        "--loss_type",
        type=str,
        default="tb",
        choices=("tb", "tb-avg", "db", "subtb", "pis", "mle"),
    )
    parser.add_argument("--subtb_lambda", type=float, default=2.0)
    parser.add_argument("--training_mode", type=str, default="fwd", choices=("fwd", "bwd", "both"))
    parser.add_argument("--bwd_from", type=str, default="buffer", choices=("energy", "buffer"))
    parser.add_argument("--clip_grad_norm", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=25000)

    parser.add_argument("--lr_policy", type=float, default=1e-3)
    parser.add_argument("--lr_Z", type=float, default=1e-1)
    parser.add_argument("--lr_flow", type=float, default=1e-2)
    parser.add_argument("--lr_back", type=float, default=None)
    parser.add_argument("--use_weight_decay", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument("--use_scheduler", action="store_true", default=False)
    parser.add_argument("--milestones", type=float, nargs="+", default=[0.5, 0.9])
    parser.add_argument("--gamma", type=float, default=(10) ** (-1 / 2))

    parser.add_argument("--hidden_dim", type=int, default=256)
    # parser.add_argument('--s_emb_dim', type=int, default=256)
    # parser.add_argument('--t_emb_dim', type=int, default=256)
    # parser.add_argument('--harmonics_dim', type=int, default=256)
    parser.add_argument("--flow_hidden_dim", type=int, default=64)
    # parser.add_argument('--flow_s_emb_dim', type=int, default=64)
    # parser.add_argument('--flow_t_emb_dim', type=int, default=64)
    # parser.add_argument('--flow_harmonics_dim', type=int, default=64)
    parser.add_argument("--lgv_layers", type=int, default=3)
    parser.add_argument("--joint_layers", type=int, default=2)
    parser.add_argument("--t_scale", type=float, default=1.0)
    parser.add_argument("--log_var_range", type=float, default=4.0)
    parser.add_argument("--lp", action="store_true", default=False)
    parser.add_argument("--lp_scaling_per_dimension", action="store_true", default=False)
    parser.add_argument("--conditional_flow_model", action="store_true", default=False)
    parser.add_argument("--share_embeddings", action="store_true", default=False)
    parser.add_argument("--learn_pb", action="store_true", default=False)
    parser.add_argument("--pb_scale_range", type=float, default=0.1)
    parser.add_argument("--learned_variance", action="store_true", default=False)
    parser.add_argument("--partial_energy", action="store_true", default=False)
    parser.add_argument("--no_clipping", action="store_false", dest="clipping")
    parser.add_argument("--lgv_clip", type=float, default=1e2)
    parser.add_argument("--gfn_clip", type=float, default=1e4)
    parser.add_argument("--no_zero_init", action="store_false", dest="zero_init")
    parser.add_argument("--no_pis_architectures", action="store_false", dest="pis_architectures")

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
    ### For local search
    parser.add_argument("--local_search", action="store_true", default=False)
    # How many iterations to run local search
    parser.add_argument("--max_iter_ls", type=int, default=200)
    # How many iterations to burn in before making local search
    parser.add_argument("--burn_in", type=int, default=100)
    # How frequently to make local search
    parser.add_argument("--ls_cycle", type=int, default=100)
    # Step size of Langevin Dynamics
    parser.add_argument("--ld_step", type=float, default=0.001)
    parser.add_argument("--ld_schedule", action="store_true", default=False)
    # Target acceptance rate
    parser.add_argument("--target_acceptance_rate", type=float, default=0.574)
    ################################################################

    ################################################################
    ### For replay buffer
    parser.add_argument("--buffer_size", type=int, default=-1)  # 100 * batch_size by default
    # prioritization
    parser.add_argument(
        "--prioritization",
        type=str,
        default="none",
        choices=("none", "reward", "loss", "normalized_iw"),
    )
    # buffer sampling strategy  # TODO: support percentile-based sampling
    parser.add_argument(
        "--buffer_sampling",
        type=str,
        default="proportional",
        choices=("proportional", "rank"),
    )
    # low rank_k give steep priorization in rank-based replay sampling
    parser.add_argument("--rank_k", type=float, default=1e-2)
    # logr_lb for filtering out samples with extremely low reward values for numerical stability
    parser.add_argument("--logr_lb", type=float, default=-1e5)
    # prefill to wait before starting to sample from buffer
    parser.add_argument(
        "--prefill", type=int, default=10
    )  # wait this amount of iterations to fill the buffer
    ################################################################

    ################################################################
    ### Exploration with extra noise
    parser.add_argument("--exploratory", action="store_true", default=False)
    parser.add_argument("--exploration_factor", type=float, default=0.1)
    parser.add_argument("--exploration_wd", action="store_true", default=False)
    ################################################################

    ################################################################
    ### Eval & Plot
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--eval_data_size", type=int, default=2000)
    parser.add_argument("--final_eval_data_size", type=int, default=2000)
    parser.add_argument("--plot_freq", type=int, default=2500)
    parser.add_argument("--plot_data_size", type=int, default=2000)
    ################################################################

    ################################################################
    ### Importance sampling related
    parser.add_argument(
        "--aux_target", type=str, default="target", choices=("target", "loss", "iw")
    )
    parser.add_argument("--train_resampling", action="store_true", default=False)
    parser.add_argument("--train_weighting", action="store_true", default=False)
    parser.add_argument("--alternating", action="store_true", default=False)
    parser.add_argument("--target_ess", type=float, default=0.0)  # 0.0 has no effect
    parser.add_argument(
        "--smoothing",
        type=str,
        default="clip_above",
        choices=("clip_above", "clip_below", "temper", "mix_with_uniform"),
    )
    parser.add_argument("--eval_resampling", action="store_true", default=False)
    parser.add_argument("--eval_weighting", action="store_true", default=False)
    parser.add_argument("--eval_buffer", action="store_true", default=False)
    ################################################################

    args = parser.parse_args()

    args.loss_type_str = args.loss_type
    if args.loss_type == "subtb":
        args.loss_type_str += f"-lambda{args.subtb_lambda}"

    set_seed(args.seed)
    if "SLURM_PROCID" in os.environ:
        args.seed += int(os.environ["SLURM_PROCID"])

    if args.lr_back is None:
        args.lr_back = args.lr_policy

    if args.buffer_size == -1:
        args.buffer_size = 100 * args.batch_size

    if args.pis_architectures:
        assert args.zero_init

    if args.loss_type in ["db", "subtb"]:
        args.conditional_flow_model = True

    if args.local_search:
        assert (
            args.training_mode == "both" or args.training_mode == "bwd"
        ) and args.bwd_from == "buffer", (
            "We only support local search for backward sampling with buffer"
        )

    assert args.plot_freq % args.eval_freq == 0

    train(args)
