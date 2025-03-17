import argparse
import os
from copy import deepcopy

import torch
import matplotlib.pyplot as plt
import wandb
from tqdm import trange

from buffer import ReplayBuffer
from energies import BaseEnergy, Funnel, ManyWell, TwentyFiveGaussianMixture
from models import GFN
from utils.eval_utils import eval_step
from utils.misc_utils import (
    cal_subtb_coef_matrix,
    get_gfn_optimizer,
    get_name,
    set_seed,
)
from utils.plot_utils import plot_step
from utils.train_utils import train_step


def get_energy(target_energy: str, device: torch.device) -> BaseEnergy:
    if target_energy == '25gmm':
        energy = TwentyFiveGaussianMixture(device=device)
    elif target_energy == 'funnel':
        energy = Funnel(device=device)
    elif target_energy == 'many_well':
        energy = ManyWell(device=device, ndim=8)
    elif target_energy == "lgcp":
        raise NotImplementedError
        energy = LGCP(device=device)
    else:
        raise ValueError(f"Unknown energy: {target_energy}")
    return energy


def train(args):
    exp_name = get_name(args)
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    config = args.__dict__
    config["Experiment"] = "{args.energy}"
    wandb.init(project="GFN Energy", config=config, name=exp_name)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    subtb_coef_matrix = None
    if args.loss_type == 'subtb':
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

    energy = get_energy(args.target_energy, device=device)
    try:
        gt_xs = energy.sample(args.eval_data_size).to(device)
    except NotImplementedError:
        gt_xs = None

    gfn_model = GFN(
        energy.ndim,
        args.harmonics_dim,
        args.t_emb_dim,
        args.s_emb_dim,
        args.hidden_dim,
        log_var_range=args.log_var_range,
        t_scale=args.t_scale,
        lp=args.lp,
        learned_variance=args.learned_variance,
        trajectory_length=args.T,
        partial_energy=args.partial_energy,
        clipping=args.clipping,
        lgv_clip=args.lgv_clip,
        gfn_clip=args.gfn_clip,
        pb_scale_range=args.pb_scale_range,
        lp_scaling_per_dimension=args.lp_scaling_per_dimension,
        conditional_flow_model=args.conditional_flow_model,
        learn_pb=args.learn_pb,
        pis_architectures=args.pis_architectures,
        lgv_layers=args.lgv_layers,
        joint_layers=args.joint_layers,
        zero_init=args.zero_init,
        device=device,
    ).to(device)

    gfn_optimizer = get_gfn_optimizer(
        gfn_model,
        args.lr_policy,
        args.lr_flow,
        args.lr_back,
        args.learn_pb,
        args.conditional_flow_model,
        args.use_weight_decay,  
        args.weight_decay,
    )

    buffer = ReplayBuffer(
        args.buffer_size,
        device,
        prioritization=args.prioritization,
        rank_k=args.rank_k,
        logr_lb=args.logr_lb,
    )
    buffer_ls = None
    if args.local_search:
        buffer_ls = deepcopy(buffer)
        buffer_ls.prioritization = "reward"

    metrics = dict()
    gfn_model.train()
    for i in trange(args.epochs + 1, dynamic_ncols=True):
        metrics['train/loss'] = train_step(
            energy,
            gfn_model,
            gfn_optimizer,
            i,
            batch_size=args.batch_size,
            loss_type=args.loss_type,
            training_mode=args.training_mode,
            clip_grad_norm=args.clip_grad_norm,
            bwd_from=args.bwd_from,
            exploratory=args.exploratory,
            exploration_factor=args.exploration_factor,
            exploration_wd=args.exploration_wd,
            buffer=buffer,
            buffer_ls=buffer_ls,
            warmup_steps=args.warmup_steps,
            local_search=args.local_search,
            ls_args=ls_args,
            subtb_coef_matrix=subtb_coef_matrix,
            device=device,
            resampling=args.train_resampling,
            weighting=args.train_weighting,
            aux_reward=args.aux_reward,
            aux_reward_temp=args.aux_reward_temp,
        )

        if i % 100 == 0:
            results, model_trajs, weights, model_trajs_r = eval_step(
                args.eval_data_size,
                gt_xs,
                gfn_model,
                energy,
                pis=args.loss_type=="pis",
                resampling=args.eval_resampling,
                weighting=args.eval_weighting,
            )
            metrics.update(results)

            images = plot_step(energy, model_trajs[:, -1])
            metrics.update(images)

            if args.eval_resampling:
                assert model_trajs_r is not None
                images_resample = plot_step(energy, model_trajs_r[:, -1])
                images_resample = {
                    k.replace("visualization/", "visualization_resample/"): v for k, v in images_resample.items()
                }
                metrics.update(images_resample)

            if args.eval_weighting:
                images_weighted = plot_step(energy, model_trajs[:, -1], weights=weights)
                images_weighted = {
                    k.replace("visualization/", "visualization_weighted/"): v for k, v in images_weighted.items()
                }
                metrics.update(images_weighted)

            plt.close('all')

            wandb.log(metrics, step=i)
            if i % 1000 == 0:
                torch.save(gfn_model.state_dict(), f'{exp_name}model.pt')

    final_results, _, _, _ = eval_step(
        args.final_eval_data_size, gt_xs, gfn_model, energy, pis=args.loss_type=="pis", final_eval=True, resampling=args.eval_resampling
    )
    metrics.update(final_results)
    wandb.log(metrics, step=args.epochs)
    torch.save(gfn_model.state_dict(), f'{exp_name}model_final.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GFN Linear Regression')
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--lr_policy', type=float, default=1e-3)
    parser.add_argument('--lr_flow', type=float, default=1e-2)
    parser.add_argument('--lr_back', type=float, default=1e-3)
    parser.add_argument('--clip_grad_norm', type=float, default=3.0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--s_emb_dim', type=int, default=64)
    parser.add_argument('--t_emb_dim', type=int, default=64)
    parser.add_argument('--harmonics_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--subtb_lambda', type=int, default=2)
    parser.add_argument('--t_scale', type=float, default=5.)
    parser.add_argument('--log_var_range', type=float, default=4.)
    parser.add_argument('--target_energy', type=str, default='9gmm', choices=('25gmm', 'funnel', 'many_well', 'lgcp'))
    parser.add_argument('--loss_type', type=str, default="tb", choices=('tb', 'tb-avg', 'db', 'subtb', "pis", "mle"))
    parser.add_argument('--training_mode', type=str, default="fwd", choices=('fwd', 'bwd', 'both'))
    parser.add_argument('--bwd_from', type=str, default="buffer", choices=('energy', 'buffer'))
    parser.add_argument('--lp', action='store_true', default=False)
    parser.add_argument('--lp_scaling_per_dimension', action='store_true', default=False)
    parser.add_argument('--conditional_flow_model', action='store_true', default=False)
    parser.add_argument('--learn_pb', action='store_true', default=False)
    parser.add_argument('--pb_scale_range', type=float, default=0.1)
    parser.add_argument('--learned_variance', action='store_true', default=False)
    parser.add_argument('--partial_energy', action='store_true', default=False)
    parser.add_argument('--exploration_factor', type=float, default=0.1)
    parser.add_argument('--exploration_wd', action='store_true', default=False)
    parser.add_argument('--clipping', action='store_true', default=False)
    parser.add_argument('--lgv_clip', type=float, default=1e2)
    parser.add_argument('--gfn_clip', type=float, default=1e4)
    parser.add_argument('--zero_init', action='store_true', default=False)
    parser.add_argument('--pis_architectures', action='store_true', default=False)
    parser.add_argument('--lgv_layers', type=int, default=3)
    parser.add_argument('--joint_layers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--weight_decay', type=float, default=1e-7)
    parser.add_argument('--use_weight_decay', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)

    parser.add_argument('--local_search', action='store_true', default=False)
    ################################################################
    ### For local search
    # How many iterations to run local search
    parser.add_argument('--max_iter_ls', type=int, default=200)
    # How many iterations to burn in before making local search
    parser.add_argument('--burn_in', type=int, default=100)
    # How frequently to make local search
    parser.add_argument('--ls_cycle', type=int, default=100)
    # Step size of Langevin Dynamics
    parser.add_argument('--ld_step', type=float, default=0.001)
    parser.add_argument('--ld_schedule', action='store_true', default=False)
    # Target acceptance rate
    parser.add_argument('--target_acceptance_rate', type=float, default=0.574)
    ################################################################

    ################################################################
    ### For replay buffer
    parser.add_argument('--buffer_size', type=int, default=100_000)
    # none -> no prioritization, rank -> rank-based prioritization; TODO: support quantile-based prioritization
    parser.add_argument('--prioritization', type=str, default="delta", choices=('none', 'reward', 'loss', 'delta'))
    # low rank_k give steep priorization in rank-based replay sampling
    parser.add_argument('--rank_k', type=float, default=1e-2)
    # logr_lb for filtering out samples with extremely low reward values
    parser.add_argument('--logr_lb', type=float, default=-10000)
    # warmup_steps to wait before starting to sample from buffer
    parser.add_argument('--warmup_steps', type=int, default=10)
    ################################################################

    parser.add_argument('--exploratory', action='store_true', default=False)

    ################################################################
    ### Eval & Plot
    parser.add_argument('--eval_data_size', type=int, default=2000)
    parser.add_argument('--final_eval_data_size', type=int, default=2000)
    parser.add_argument('--plot_data_size', type=int, default=2000)
    ################################################################

    ################################################################
    ### Importance sampling related
    parser.add_argument('--train_resampling', action='store_true', default=False)
    parser.add_argument('--train_weighting', action='store_true', default=False)
    parser.add_argument('--aux_reward', type=str, default="reward", choices=("reward", "loss", "delta"))
    parser.add_argument('--aux_reward_temp', type=float, default=1.0)
    parser.add_argument('--eval_resampling', action='store_true', default=False)
    parser.add_argument('--eval_weighting', action='store_true', default=False)
    ################################################################

    args = parser.parse_args()

    set_seed(args.seed)
    if 'SLURM_PROCID' in os.environ:
        args.seed += int(os.environ["SLURM_PROCID"])

    if args.pis_architectures:
        args.zero_init = True

    if args.local_search:
        assert (
            (args.training_mode == "both" or args.training_mode == "bwd")
            and args.bwd_from == "buffer"
        ), "We only support local search for backward sampling with buffer"

    if "tb" not in args.loss_type and args.prioritization == "delta":
        raise ValueError("Delta prioritization is only supported for tb loss")

    train(args)
