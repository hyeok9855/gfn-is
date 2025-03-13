import argparse
import os
from copy import deepcopy

import torch
import matplotlib.pyplot as plt
import wandb
from tqdm import trange

from buffer import ReplayBuffer
from energies import BaseEnergy, Funnel, ManyWell, TwentyFiveGaussianMixture
from langevin import langevin_dynamics
from models import GFN
from utils.eval_utils import eval_step
from utils.misc_utils import (
    cal_subtb_coef_matrix,
    get_exploration_std,
    get_gfn_optimizer,
    get_gfn_forward_loss,
    get_gfn_backward_loss,
    get_name,
    set_seed,
)
from utils.plot_utils import plot_step


parser = argparse.ArgumentParser(description='GFN Linear Regression')
parser.add_argument('--lr_policy', type=float, default=1e-3)
parser.add_argument('--lr_flow', type=float, default=1e-2)
parser.add_argument('--lr_back', type=float, default=1e-3)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--s_emb_dim', type=int, default=64)
parser.add_argument('--t_emb_dim', type=int, default=64)
parser.add_argument('--harmonics_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--epochs', type=int, default=25000)
parser.add_argument('--buffer_size', type=int, default=300 * 1000 * 2)
parser.add_argument('--T', type=int, default=100)
parser.add_argument('--subtb_lambda', type=int, default=2)
parser.add_argument('--t_scale', type=float, default=5.)
parser.add_argument('--log_var_range', type=float, default=4.)
parser.add_argument('--energy', type=str, default='9gmm', choices=('25gmm', 'funnel', 'many_well', 'lgcp'))
parser.add_argument('--mode_fwd', type=str, default="tb", choices=('tb', 'tb-avg', 'db', 'subtb', "pis"))
parser.add_argument('--mode_bwd', type=str, default="tb", choices=('tb', 'tb-avg', 'mle'))
parser.add_argument('--both_ways', action='store_true', default=False)

# For local search
################################################################
parser.add_argument('--local_search', action='store_true', default=False)

# How many iterations to run local search
parser.add_argument('--max_iter_ls', type=int, default=200)

# How many iterations to burn in before making local search
parser.add_argument('--burn_in', type=int, default=100)

# How frequently to make local search
parser.add_argument('--ls_cycle', type=int, default=100)

# langevin step size
parser.add_argument('--ld_step', type=float, default=0.001)

parser.add_argument('--ld_schedule', action='store_true', default=False)

# target acceptance rate
parser.add_argument('--target_acceptance_rate', type=float, default=0.574)


# For replay buffer
################################################################
# high beta give steep priorization in reward prioritized replay sampling
parser.add_argument('--beta', type=float, default=1.)

# low rank_weighted give steep priorization in rank-based replay sampling
parser.add_argument('--rank_weight', type=float, default=1e-2)

# three kinds of replay training: random, reward prioritized, rank-based
parser.add_argument('--prioritized', type=str, default="rank", choices=('none', 'reward', 'rank'))
################################################################

parser.add_argument('--bwd', action='store_true', default=False)
parser.add_argument('--exploratory', action='store_true', default=False)

parser.add_argument('--sampling', type=str, default="buffer", choices=('energy', 'buffer'))
parser.add_argument('--langevin', action='store_true', default=False)
parser.add_argument('--langevin_scaling_per_dimension', action='store_true', default=False)
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

################################################################
# Resampling
parser.add_argument('--eval_resample', action='store_true', default=False)
################################################################

args = parser.parse_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

eval_data_size = 2000
final_eval_data_size = 2000
plot_data_size = 2000

if args.pis_architectures:
    args.zero_init = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
coeff_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.T).to(device)

if args.both_ways and args.bwd:
    args.bwd = False

if args.local_search:
    args.both_ways = True


def get_energy():
    if args.energy == '25gmm':
        energy = TwentyFiveGaussianMixture(device=device)
    elif args.energy == 'funnel':
        energy = Funnel(device=device)
    elif args.energy == 'many_well':
        energy = ManyWell(device=device)
    elif args.energy == "lgcp":
        raise NotImplementedError
        energy = LGCP(device=device)
    else:
        raise ValueError(f"Unknown energy: {args.energy}")
    return energy


def train_step(
    energy: BaseEnergy,
    gfn_model: GFN,
    gfn_optimizer: torch.optim.Optimizer,
    it: int,
    exploratory,
    buffer,
    buffer_ls,
    exploration_factor,
    exploration_wd,
):
    gfn_model.zero_grad()

    exploration_std = get_exploration_std(it, exploratory, exploration_factor, exploration_wd)

    if args.both_ways:
        if it % 2 == 0:
            if args.sampling == 'buffer':
                loss, states, _, _, log_r  = fwd_train_step(energy, gfn_model, exploration_std, return_exp=True)
                buffer.add(states[:, -1],log_r)
            else:
                loss = fwd_train_step(energy, gfn_model, exploration_std)
        else:
            loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, it=it)

    elif args.bwd:
        loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, it=it)
    else:
        loss = fwd_train_step(energy, gfn_model, exploration_std)

    loss.backward()
    gfn_optimizer.step()
    return loss.item()


def fwd_train_step(energy: BaseEnergy, gfn_model: GFN, exploration_std=0.0, return_exp=False):
    init_state = torch.zeros(args.batch_size, energy.ndim).to(device)
    loss = get_gfn_forward_loss(
        args.mode_fwd,
        init_state,
        gfn_model,
        energy.log_reward,
        coeff_matrix,
        exploration_std=exploration_std,
        return_exp=return_exp,
    )
    return loss


def bwd_train_step(energy: BaseEnergy, gfn_model: GFN, buffer: ReplayBuffer, buffer_ls: ReplayBuffer, it=0):
    if args.sampling == 'energy':
        samples = energy.sample(args.batch_size).to(device)
    elif args.sampling == 'buffer':
        if args.local_search:
            if it % args.ls_cycle < 2:
                samples, _ = buffer.sample()
                local_search_samples, log_r = langevin_dynamics(samples, energy.log_reward, device, args)
                buffer_ls.add(local_search_samples, log_r)
            samples, _ = buffer_ls.sample()
        else:
            samples, _ = buffer.sample()

    loss = get_gfn_backward_loss(args.mode_bwd, samples, gfn_model, energy.log_reward)
    return loss


def train():
    name = get_name(args)
    if not os.path.exists(name):
        os.makedirs(name)

    energy = get_energy()
    try:
        gt_xs = energy.sample(eval_data_size).to(device)
    except NotImplementedError:
        gt_xs = None

    config = args.__dict__
    config["Experiment"] = "{args.energy}"
    wandb.init(project="GFN Energy", config=config, name=name)

    gfn_model = GFN(
        energy.ndim,
        args.harmonics_dim,
        args.t_emb_dim,
        args.s_emb_dim,
        args.hidden_dim,
        log_var_range=args.log_var_range,
        t_scale=args.t_scale,
        langevin=args.langevin,
        learned_variance=args.learned_variance,
        trajectory_length=args.T,
        partial_energy=args.partial_energy,
        clipping=args.clipping,
        lgv_clip=args.lgv_clip,
        gfn_clip=args.gfn_clip,
        pb_scale_range=args.pb_scale_range,
        langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
        conditional_flow_model=args.conditional_flow_model,
        learn_pb=args.learn_pb,
        pis_architectures=args.pis_architectures,
        lgv_layers=args.lgv_layers,
        joint_layers=args.joint_layers,
        zero_init=args.zero_init,
        device=device
    ).to(device)


    gfn_optimizer = get_gfn_optimizer(gfn_model, args.lr_policy, args.lr_flow, args.lr_back, args.learn_pb,
                                      args.conditional_flow_model, args.use_weight_decay, args.weight_decay)

    print(gfn_model)
    metrics = dict()

    buffer = ReplayBuffer(
        args.buffer_size,
        device,
        energy.log_reward,
        args.batch_size,
        data_ndim=energy.ndim,
        beta=args.beta,
        rank_weight=args.rank_weight,
        prioritized=args.prioritized,
    )
    buffer_ls = deepcopy(buffer)

    gfn_model.train()
    for i in trange(args.epochs + 1):
        metrics['train/loss'] = train_step(
            energy,
            gfn_model,
            gfn_optimizer,
            i,
            args.exploratory,
            buffer,
            buffer_ls,
            args.exploration_factor,
            args.exploration_wd
        )

        if i % 100 == 0:
            results, model_trajs, model_trajs_r = eval_step(
                eval_data_size, gt_xs, gfn_model, energy, pis=args.mode_fwd=="pis", resample=args.eval_resample
            )
            metrics.update(results)

            images = plot_step(energy, model_trajs[:, -1], gt_xs, plot_data_size, device)
            metrics.update(images)
            if args.eval_resample:
                assert model_trajs_r is not None
                images_resample = plot_step(energy, model_trajs_r[:, -1], gt_xs, plot_data_size, device)
                images_resample = {
                    k.replace("visualization/", "visualization_resample/"): v for k, v in images_resample.items()
                }
                metrics.update(images_resample)
            plt.close('all')

            wandb.log(metrics, step=i)
            if i % 1000 == 0:
                torch.save(gfn_model.state_dict(), f'{name}model.pt')

    final_results, _, _ = eval_step(
        final_eval_data_size, gt_xs, gfn_model, energy, pis=args.mode_fwd=="pis", final_eval=True, resample=args.eval_resample
    )
    metrics.update(final_results)
    wandb.log(metrics, step=args.epochs)
    torch.save(gfn_model.state_dict(), f'{name}model_final.pt')


# def eval():
#     name = get_name(args)

#     print(name)

#     energy = get_energy()
#     eval_data = energy.sample(eval_data_size).to(device) if not (args.energy in _LIST_OF_NO_SAMPLES_ENERGIES) else None

#     gfn_model = GFN(energy.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
#                     clipping=args.clipping, lgv_clip=args.lgv_clip, gfn_clip=args.gfn_clip,
#                     langevin=args.langevin, learned_variance=args.learned_variance,
#                     partial_energy=args.partial_energy, log_var_range=args.log_var_range,
#                     pb_scale_range=args.pb_scale_range,
#                     t_scale=args.t_scale, langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
#                     conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
#                     pis_architectures=args.pis_architectures, lgv_layers=args.lgv_layers,
#                     joint_layers=args.joint_layers, zero_init=args.zero_init, device=device).to(device)

#     model_final_path = name + 'model_final.pt'
#     model_path = name + 'model.pt'

#     if os.path.exists(model_final_path):
#         try:
#             gfn_model.load_state_dict(torch.load(model_final_path, weights_only=True))
#         except:
#             print("Couldn't load final model")
#     else:
#         if os.path.exists(model_path):
#             try:
#                 gfn_model.load_state_dict(torch.load(model_path, weights_only=True))
#             except:
#                 print("Couldn't load model")
#         else:
#             print("NO MODEL IS AVAILABLE")
#             return

#     config = args.__dict__
#     config["Experiment"] = "{args.energy}"
#     wandb.init(project="GFN Energy - proper evaluation", config=config, name=name)

#     print(gfn_model)
#     metrics = dict()

#     gfn_model.eval()
#     for i in trange(1, 201):
#         metrics.update(eval_step_K_step_discretizer(eval_data, energy, gfn_model, final_eval=False, traj_length=i))
#         # if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
#         #     del metrics[f'eval_{i}_steps/log_Z_learned']
#     wandb.log(metrics)


if __name__ == '__main__':
    # if args.eval:
    #     eval()
    # else:
    train()
