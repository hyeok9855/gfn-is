import itertools

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


def plot_contours(
    log_prob, device, ax=None, bounds=(-10.0, 10.0), grid_width_n_points=200, n_contour_levels=50, clamp_min=-1000.0
):
    """Plot contours of a log_prob_func that is defined on 2D"""
    if ax is None:
        fig, ax = plt.subplots(1)
    x_points_dim1 = torch.linspace(bounds[0], bounds[1], grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_p_x = log_prob(x_points.to(device)).detach().cpu()
    log_p_x = torch.clamp_min(log_p_x, clamp_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
    x_points_dim1 = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    x_points_dim2 = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    if n_contour_levels:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x, levels=n_contour_levels)
    else:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x)


def plot_samples(samples, ax=None, bounds=(-10.0, 10.0), alpha=0.5):
    if ax is None:
        fig, ax = plt.subplots(1)
    samples = torch.clamp(samples, bounds[0], bounds[1])
    samples = samples.cpu().detach()
    ax.scatter(samples[:, 0], samples[:, 1], alpha=alpha, marker="o", s=10)


def plot_kde(samples, ax=None, bounds=(-10.0, 10.0)):
    if ax is None:
        fig, ax = plt.subplots(1)
    samples = samples.cpu().detach()
    sns.kdeplot(x=samples[:, 0], y=samples[:, 1], cmap="Blues", fill=True, ax=ax, clip=bounds)


def viz_2d_slice(
    energy: BaseEnergy,
    dims: tuple,
    samples: torch.Tensor,
    lim=3.0,
    alpha=0.8,
    n_contour_levels=50,
    clamp_min=-1000.0,
    kde=True,
):
    dim_str = "".join(map(lambda _d: str(_d + 1), dims))

    x = samples[:, dims].detach().cpu()

    if kde:
        fig_kde, ax_kde = viz_kde2d(x, "kde", f"kde{dim_str}.png", lim=lim)
    else:
        fig_kde, ax_kde = None, None

    def logp_func(x_2d):
        _x = torch.zeros((x_2d.shape[0], energy.ndim))
        _x[:, dims] = x_2d
        return energy.log_reward(_x.to(energy.device))

    fig_contour, ax_contour = viz_contour_sample2d(
        x,
        f"contour{dim_str}.png",
        logp_func,
        lim=lim,
        alpha=alpha,
        n_contour_levels=n_contour_levels,
        clamp_min=clamp_min,
    )

    return fig_kde, ax_kde, fig_contour, ax_contour


def viz_manywell(energy: ManyWell, samples: torch.Tensor) -> list:
    lim = energy.plot_bound
    out = []
    out.extend(viz_2d_slice(energy, (0, 2), samples, lim=lim))
    out.extend(viz_2d_slice(energy, (1, 2), samples, lim=lim))
    out.extend(viz_2d_slice(energy, (0, 4), samples, lim=lim))
    out.extend(viz_2d_slice(energy, (1, 4), samples, lim=lim))
    return out


def viz_funnel(energy: Funnel, samples: torch.Tensor) -> list:
    lim = energy.plot_bound
    out = []
    for i in range(1, 5):
        out.extend(viz_2d_slice(energy, (0, i), samples, lim=lim, kde=False))
    return out


def viz_gmm(
    energy: TwentyFiveGaussianMixture, samples: torch.Tensor, n_contour_levels=100, clamp_min=-100000.0
) -> list:
    lim = energy.plot_bound
    out = []
    for i in range(0, min(energy.ndim, 8), 2):
        out.extend(
            viz_2d_slice(
                energy, (i, i + 1), samples, lim=lim, n_contour_levels=n_contour_levels, clamp_min=clamp_min, kde=False
            )
        )
    return out


def traj_plot1d(traj_len, samples, xlabel, ylabel, title="", fsave="img.png"):
    samples = rearrange(samples, "t b d -> b t d").cpu()
    inds = np.linspace(0, samples.shape[1], traj_len, endpoint=False, dtype=int)
    samples = samples[:, inds]
    plt.figure()
    for i, sample in enumerate(samples):
        plt.plot(np.arange(traj_len), sample.flatten(), marker="x", label=f"sample {i}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fsave)
    plt.close()


########### 2D plot
def viz_sample2d(points, title, fsave, lim=7.0, sample_num=50000):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    if title is not None:
        ax.set_title(title)
    ax.plot(
        points[:sample_num, 0],
        points[:sample_num, 1],
        linewidth=0,
        marker=".",
        markersize=1,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    return fig, ax


def viz_kde2d(points, title, fname, lim=7.0, sample_num=2000):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=200)
    if title is not None:
        ax.set_title(title)
    try:
        sns.kdeplot(x=points[:sample_num, 0], y=points[:sample_num, 1], cmap="coolwarm", fill=True, ax=ax)
    except:
        print("Error in kde plot")
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


def viz_contour_sample2d(points, fname, log_prob_func, lim=3.0, alpha=0.7, n_contour_levels=None, clamp_min=-1000.0):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    viz_contour_with_ax(ax, log_prob_func, lim=lim, n_contour_levels=n_contour_levels, clamp_min=clamp_min)

    samples = torch.clamp(points, -lim, lim)
    samples = samples.cpu().detach()
    ax.plot(samples[:, 0], samples[:, 1], linewidth=0, marker=".", markersize=1.5, alpha=alpha)

    return fig, ax


def plot_step(
    energy: BaseEnergy,
    samples: torch.Tensor,
    gt_samples: torch.Tensor | None = None,
    plot_batch_size=5000,
    device=torch.device("cpu"),
):
    if isinstance(energy, ManyWell):
        vizualizations = viz_manywell(energy, samples)
        fig_kde_x02, fig_contour_x02, fig_kde_x12, fig_contour_x12 = vizualizations[0:8:2]
        fig_kde_x04, fig_contour_x04, fig_kde_x14, fig_contour_x14 = vizualizations[8::2]

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

    elif isinstance(energy, Funnel):
        vizualizations = viz_funnel(energy, samples)

        out_dict = {}
        for i in range(1, 5):
            fig_kde, ax_kde, fig_contour, ax_contour = vizualizations[4 * (i - 1) : 4 * i]
            out_dict.update(
                {
                    f"visualization/contour0{i}": wandb.Image(fig_to_image(fig_contour)),
                    # f"visualization/kde0{i}": wandb.Image(fig_to_image(fig_kde)),
                }
            )

    elif isinstance(energy, TwentyFiveGaussianMixture):
        vizualizations = viz_gmm(energy, samples)

        out_dict = {}
        for i in range(len(vizualizations) // 4):
            fig_kde, ax_kde, fig_contour, ax_contour = vizualizations[4 * i : 4 * (i + 1)]
            out_dict.update(
                {
                    f"visualization/contour{2 * i}{2 * i + 1}": wandb.Image(fig_to_image(fig_contour)),
                    # f"visualization/kde{2 * i}{2 * i + 1}": wandb.Image(fig_to_image(fig_kde)),
                }
            )

    elif energy.ndim != 2:
        out_dict = {}

    else:
        if gt_samples is None:
            gt_samples = energy.sample(plot_batch_size)

        lim = getattr(energy, "_plot_bound", gt_samples.abs().max().item() * 1.1)

        fig_contour, ax_contour = get_figure(bounds=(-lim, lim))
        fig_kde, ax_kde = get_figure(bounds=(-lim, lim))
        fig_kde_overlay, ax_kde_overlay = get_figure(bounds=(-lim, lim))

        plot_contours(energy.log_reward, device=device, ax=ax_contour, bounds=(-lim, lim), n_contour_levels=150)
        plot_kde(gt_samples, ax=ax_kde_overlay, bounds=(-lim, lim))
        plot_kde(samples, ax=ax_kde, bounds=(-lim, lim))
        plot_samples(samples, ax=ax_contour, bounds=(-lim, lim))
        plot_samples(samples, ax=ax_kde_overlay, bounds=(-lim, lim))

        out_dict = {
            "visualization/contour": wandb.Image(fig_to_image(fig_contour)),
            "visualization/kde_overlay": wandb.Image(fig_to_image(fig_kde_overlay)),
            "visualization/kde": wandb.Image(fig_to_image(fig_kde)),
        }

    return out_dict
