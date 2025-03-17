import itertools
from typing import Callable

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import PIL
import seaborn as sns
import torch
import wandb
from einops import rearrange

from energies import BaseEnergy, Funnel, ManyWell, TwentyFiveGaussianMixture


def get_figure(bounds=(-10.0, 10.0)):
    fig, ax = plt.subplots(1, figsize=(16, 16))
    ax.axis("off")
    ax.set_autoscale_on(False)
    ax.set_xlim([bounds[0], bounds[1]])
    ax.set_ylim([bounds[0], bounds[1]])
    return fig, ax


def fig_to_image(fig):
    fig.canvas.draw()
    return PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())  # type: ignore


# def plot_contours(
#     log_prob, device, ax=None, bounds=(-10.0, 10.0), grid_width_n_points=200, n_contour_levels=50, clamp_min=-1000.0
# ):
#     """Plot contours of a log_prob_func that is defined on 2D"""
#     if ax is None:
#         fig, ax = plt.subplots(1)
#     x_points_dim1 = torch.linspace(bounds[0], bounds[1], grid_width_n_points)
#     x_points_dim2 = x_points_dim1
#     x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
#     log_p_x = log_prob(x_points.to(device)).detach().cpu()
#     log_p_x = torch.clamp_min(log_p_x, clamp_min)
#     log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
#     x_points_dim1 = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).numpy()
#     x_points_dim2 = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).numpy()
#     if n_contour_levels:
#         ax.contour(x_points_dim1, x_points_dim2, log_p_x, levels=n_contour_levels)
#     else:
#         ax.contour(x_points_dim1, x_points_dim2, log_p_x)


# def plot_samples(samples, ax=None, bounds=(-10.0, 10.0), alpha=0.5):
#     if ax is None:
#         fig, ax = plt.subplots(1)
#     samples = torch.clamp(samples, bounds[0], bounds[1])
#     samples = samples.cpu().detach()
#     ax.scatter(samples[:, 0], samples[:, 1], alpha=alpha, marker="o", s=10)


# def plot_kde(
#     samples: torch.Tensor,
#     weights: torch.Tensor | None = None,
#     ax=None,
#     bounds=(-10.0, 10.0),
# ):
#     if ax is None:
#         fig, ax = plt.subplots(1)
#     samples = samples.cpu().detach()

#     if weights is not None:
#         assert samples.shape[0] == weights.shape[0]

#     sns.kdeplot(
#         x=samples[:, 0],
#         y=samples[:, 1],
#         weights=weights,
#         cmap="Blues",
#         fill=True,
#         ax=ax,
#         clip=bounds,
#     )


def viz_2d_slice(
    energy: BaseEnergy,
    dims: tuple,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
    lim=3.0,
    alpha=0.8,
    n_contour_levels=50,
    clamp_min=-1000.0,
    kde=True,
):
    x = samples[:, dims].detach().cpu()
    weights = weights.detach().cpu() if weights is not None else None

    if kde:
        fig_kde, ax_kde = viz_kde2d(
            x, "kde", weights=weights, lim=lim
        )
    else:
        fig_kde, ax_kde = None, None

    def logp_func(x_2d):
        _x = torch.zeros((x_2d.shape[0], energy.ndim))
        _x[:, dims] = x_2d
        return energy.log_reward(_x.to(energy.device))

    fig_contour, ax_contour = viz_contour_sample2d(
        x,
        logp_func,
        weights=weights,
        lim=lim,
        alpha=alpha,
        n_contour_levels=n_contour_levels,
        clamp_min=clamp_min,
    )

    return fig_kde, ax_kde, fig_contour, ax_contour


def viz_manywell(
    energy: ManyWell,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> dict:
    lim = energy.plot_bound
    viz_lst = []
    viz_lst.extend(viz_2d_slice(energy, (0, 2), samples, weights=weights, lim=lim))
    viz_lst.extend(viz_2d_slice(energy, (1, 2), samples, weights=weights, lim=lim))
    viz_lst.extend(viz_2d_slice(energy, (0, 4), samples, weights=weights, lim=lim))
    viz_lst.extend(viz_2d_slice(energy, (1, 4), samples, weights=weights, lim=lim))

    fig_kde_x02, fig_contour_x02, fig_kde_x12, fig_contour_x12 = viz_lst[0:8:2]
    fig_kde_x04, fig_contour_x04, fig_kde_x14, fig_contour_x14 = viz_lst[8::2]

    out_dict = {
        "visualization/contourx02": wandb.Image(fig_to_image(fig_contour_x02)),
        "visualization/contourx12": wandb.Image(fig_to_image(fig_contour_x12)),
        "visualization/kdex02": wandb.Image(fig_to_image(fig_kde_x02)),
        "visualization/kdex12": wandb.Image(fig_to_image(fig_kde_x12)),
        "visualization/contourx04": wandb.Image(fig_to_image(fig_contour_x04)),
        "visualization/contourx14": wandb.Image(fig_to_image(fig_contour_x14)),
        "visualization/kdex04": wandb.Image(fig_to_image(fig_kde_x04)),
        "visualization/kdex14": wandb.Image(fig_to_image(fig_kde_x14)),
    }

    for obj in viz_lst:
        if isinstance(obj, Figure):
            plt.close(obj)

    return out_dict


def viz_funnel(
    energy: Funnel,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> dict:
    lim = energy.plot_bound
    viz_lst = []
    for i in range(1, 5):
        viz_lst.extend(viz_2d_slice(energy, (0, i), samples, weights=weights, lim=lim))

    out_dict = {}
    for i in range(1, 5):
        fig_kde, ax_kde, fig_contour, ax_contour = viz_lst[4 * (i - 1) : 4 * i]
        out_dict.update(
            {
                f"visualization/contour0{i}": wandb.Image(fig_to_image(fig_contour)),
                f"visualization/kde0{i}": wandb.Image(fig_to_image(fig_kde)),
            }
        )
        plt.close(fig_kde)
        plt.close(fig_contour)

    return out_dict


def viz_gmm(
    energy: TwentyFiveGaussianMixture,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
    n_contour_levels=10,
    clamp_min=-100000.0,
) -> dict:
    lim = energy.plot_bound
    viz_lst = []
    for i in range(0, min(energy.ndim, 8), 2):
        viz_lst.extend(
            viz_2d_slice(
                energy, (i, i + 1),
                samples,
                weights=weights,
                lim=lim,
                n_contour_levels=n_contour_levels,
                clamp_min=clamp_min,
            )
        )

    out_dict = {}
    for i in range(len(viz_lst) // 4):
        fig_kde, ax_kde, fig_contour, ax_contour = viz_lst[4 * i : 4 * (i + 1)]
        out_dict.update(
            {
                f"visualization/contour{2 * i}{2 * i + 1}": wandb.Image(fig_to_image(fig_contour)),
                f"visualization/kde{2 * i}{2 * i + 1}": wandb.Image(fig_to_image(fig_kde)),
            }
        )
        plt.close(fig_kde)
        plt.close(fig_contour)

    return out_dict


def viz_kde2d(
    points: torch.Tensor,
    title: str,
    weights: torch.Tensor | None = None,
    lim=7.0,
    sample_num=2000,
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=200)
    if title is not None:
        ax.set_title(title)
    try:
        sns.kdeplot(
            x=points[:sample_num, 0],
            y=points[:sample_num, 1],
            weights=weights[:sample_num] if weights is not None else None,
            cmap="coolwarm",
            fill=True,
            ax=ax,
            warn_singular=False,
        )
    except Exception as e:
        print(f"Error in kde plot: {e}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    return fig, ax


def viz_contour_with_ax(ax, log_prob_func, lim=3.0, n_contour_levels=None, clamp_min=-1000.0):
    grid_width_n_points = 100
    x_points_dim1 = torch.linspace(-lim, lim, grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_p_x = log_prob_func(x_points).detach().cpu()
    log_p_x = torch.clamp_min(log_p_x, clamp_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
    x_points_dim1 = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    x_points_dim2 = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    if n_contour_levels:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x, levels=n_contour_levels)
    else:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x)


def viz_contour_sample2d(
    points: torch.Tensor,
    log_prob_func: Callable,
    weights: torch.Tensor | None = None,
    lim=3.0,
    alpha=0.7,
    n_contour_levels=None,
    clamp_min=-1000.0,
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    viz_contour_with_ax(
        ax, log_prob_func, lim=lim, n_contour_levels=n_contour_levels, clamp_min=clamp_min
    )

    samples = torch.clamp(points, -lim, lim)

    # ax.plot(
    #     samples[:, 0],
    #     samples[:, 1],
    #     linewidth=0,
    #     marker=".",
    #     markersize=1.5,
    #     alpha=alpha,
    # )

    # weights are used for the size of the markers
    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        alpha=alpha,
        marker="o",
        s=weights[: len(samples)] * len(samples) * 5 if weights is not None else 5,
    )

    return fig, ax


def plot_step(
    energy: BaseEnergy,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
    # gt_samples: torch.Tensor | None = None,
    # plot_batch_size=5000,
    # device=torch.device("cpu"),
):
    if isinstance(energy, ManyWell):
        out_dict = viz_manywell(energy, samples, weights)

    elif isinstance(energy, Funnel):
        out_dict = viz_funnel(energy, samples, weights)

    elif isinstance(energy, TwentyFiveGaussianMixture):
        out_dict = viz_gmm(energy, samples, weights)

    elif energy.ndim != 2:
        out_dict = {}

    else:
        raise NotImplementedError
        # if gt_samples is None:
        #     gt_samples = energy.sample(plot_batch_size)

        # lim = getattr(energy, "_plot_bound", gt_samples.abs().max().item() * 1.1)

        # fig_contour, ax_contour = get_figure(bounds=(-lim, lim))
        # fig_kde, ax_kde = get_figure(bounds=(-lim, lim))
        # fig_kde_overlay, ax_kde_overlay = get_figure(bounds=(-lim, lim))

        # plot_contours(energy.log_reward, device=device, ax=ax_contour, bounds=(-lim, lim), n_contour_levels=150)
        # plot_kde(gt_samples, ax=ax_kde_overlay, bounds=(-lim, lim))
        # plot_kde(samples, weights=weights, ax=ax_kde, bounds=(-lim, lim))
        # plot_samples(samples, ax=ax_contour, bounds=(-lim, lim))
        # plot_samples(samples, ax=ax_kde_overlay, bounds=(-lim, lim))

        # out_dict = {
        #     "visualization/contour": wandb.Image(fig_to_image(fig_contour)),
        #     "visualization/kde_overlay": wandb.Image(fig_to_image(fig_kde_overlay)),
        #     "visualization/kde": wandb.Image(fig_to_image(fig_kde)),
        # }

    return out_dict
