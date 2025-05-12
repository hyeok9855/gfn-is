import itertools
import warnings
from typing import Callable

import matplotlib.pyplot as plt
import PIL
import seaborn as sns
import torch
import wandb
from matplotlib.figure import Figure

from energies import (
    GMM40,
    BaseEnergy,
    Funnel,
    IntermediateEnergy,
    LennardJones,
    ManyWell,
    TwentyFiveGaussianMixture,
)
from utils.particle_system import interatomic_distance


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


def viz_kde2d(
    points: torch.Tensor,
    title: str,
    weights: torch.Tensor | None = None,
    lim=3.0,
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=200)
    if title is not None:
        ax.set_title(title)
    try:
        sns.kdeplot(
            x=points[:, 0],
            y=points[:, 1],
            weights=weights if weights is not None else None,
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


def viz_contour_with_ax(
    ax: plt.Axes,
    log_prob_func: Callable[[torch.Tensor], torch.Tensor],
    lim=3.0,
    n_contour_levels=50,
    grid_width_n_points=200,
    clamp_min=-1000.0,
    zorder=1,
):
    x_points_dim1 = torch.linspace(-lim, lim, grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_p_x = log_prob_func(x_points).detach().cpu()
    log_p_x = torch.clamp_min(log_p_x, clamp_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
    x_points_dim1 = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    x_points_dim2 = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    ax.contour(x_points_dim1, x_points_dim2, log_p_x, levels=n_contour_levels, zorder=zorder)


def viz_contour_sample2d(
    points: torch.Tensor,
    log_prob_func: Callable[[torch.Tensor], torch.Tensor],
    weights: torch.Tensor | None = None,
    lim=3.0,
    alpha=0.7,
    n_contour_levels=50,
    grid_width_n_points=200,
    clamp_min=-1000.0,
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    viz_contour_with_ax(
        ax,
        log_prob_func,
        lim=lim,
        n_contour_levels=n_contour_levels,
        grid_width_n_points=grid_width_n_points,
        clamp_min=clamp_min,
        zorder=2,
    )

    samples = torch.clamp(points, -lim, lim)

    # weights are used for the size of the markers
    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        alpha=alpha,
        marker="o",
        s=weights[: len(samples)] * len(samples) * 5 if weights is not None else 5,
        zorder=1,
    )

    return fig, ax


def viz_2d_slice(
    energy: BaseEnergy,
    dims: tuple,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
    lim=3.0,
    alpha=0.8,
    n_contour_levels=50,
    grid_width_n_points=200,
    clamp_min=-1000.0,
    kde=True,
):
    x = samples[:, dims].detach().cpu()
    weights = weights.detach().cpu() if weights is not None else None

    if kde:
        fig_kde, ax_kde = viz_kde2d(x, "kde", weights=weights, lim=lim)
    else:
        fig_kde, ax_kde = None, None

    def logp_func(x_2d: torch.Tensor) -> torch.Tensor:
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
        grid_width_n_points=grid_width_n_points,
        clamp_min=clamp_min,
    )

    return fig_kde, ax_kde, fig_contour, ax_contour


def viz_manywell(
    energy: BaseEnergy,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> dict:
    lim = energy.plot_bound
    viz_lst = []
    viz_lst.extend(viz_2d_slice(energy, (0, 2), samples, weights=weights, lim=lim))
    viz_lst.extend(viz_2d_slice(energy, (1, 2), samples, weights=weights, lim=lim))
    # viz_lst.extend(viz_2d_slice(energy, (4, 6), samples, weights=weights, lim=lim))
    # viz_lst.extend(viz_2d_slice(energy, (5, 6), samples, weights=weights, lim=lim))

    fig_kde_x02, fig_contour_x02, fig_kde_x12, fig_contour_x12 = viz_lst[0:8:2]
    # fig_kde_x46, fig_contour_x46, fig_kde_x56, fig_contour_x56 = viz_lst[8::2]

    out_dict = {
        "visualization/contourx02": wandb.Image(fig_to_image(fig_contour_x02)),
        "visualization/contourx12": wandb.Image(fig_to_image(fig_contour_x12)),
        "visualization/kdex02": wandb.Image(fig_to_image(fig_kde_x02)),
        "visualization/kdex12": wandb.Image(fig_to_image(fig_kde_x12)),
        # "visualization/contourx46": wandb.Image(fig_to_image(fig_contour_x46)),
        # "visualization/contourx56": wandb.Image(fig_to_image(fig_contour_x56)),
        # "visualization/kdex46": wandb.Image(fig_to_image(fig_kde_x46)),
        # "visualization/kdex56": wandb.Image(fig_to_image(fig_kde_x56)),
    }

    for obj in viz_lst:
        if isinstance(obj, Figure):
            plt.close(obj)

    return out_dict


def viz_funnel(
    energy: BaseEnergy,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> dict:
    lim = energy.plot_bound
    viz_lst = []
    for i in range(1, 3):
        viz_lst.extend(viz_2d_slice(energy, (0, i), samples, weights=weights, lim=lim))

    out_dict = {}
    for i in range(1, 3):
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
    energy: BaseEnergy,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
    clamp_min=-1000.0,
) -> dict:
    lim = energy.plot_bound
    viz_lst = []
    for i in range(1, min(energy.ndim, 4), 2):
        viz_lst.extend(
            viz_2d_slice(
                energy,
                (i - 1, i),
                samples,
                weights=weights,
                lim=lim,
                clamp_min=clamp_min,
                n_contour_levels=100,
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


def viz_lennard_jones(energy: LennardJones, xs: torch.Tensor) -> dict:
    xs_logr = energy.log_reward(xs)
    gt_xs, gt_xs_logr = energy.cached_sample(xs.shape[0])

    xs, gt_xs = xs.cpu(), gt_xs.cpu()
    xs_logr, gt_xs_logr = xs_logr.cpu(), gt_xs_logr.cpu()

    dist_xs = interatomic_distance(xs, energy.n_particles, energy.spatial_dim).view(-1)
    dist_gt_xs = interatomic_distance(gt_xs, energy.n_particles, energy.spatial_dim).view(-1)

    if energy.n_particles == 13:
        bins = 100
        min_dist = 0
        max_dist = 6
        min_energy = -60
        max_energy = 0
    elif energy.n_particles == 55:
        bins = 200
        min_dist = 0
        max_dist = 6
        min_energy = -380
        max_energy = -180
    else:
        raise ValueError(f"Unknown number of particles: {energy.n_particles}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for _xs, color in zip([dist_xs, dist_gt_xs], ["r", "g"]):
        axes[0].hist(
            torch.clamp(_xs, min=min_dist, max=max_dist),
            bins=bins,
            alpha=0.5,
            density=True,
            histtype="step",
            color=color,
            linewidth=4,
        )
    axes[0].set_xlabel("Interatomic Distances")
    axes[0].legend(["generated data", "test data"])
    axes[0].grid(True)

    for _logr, color in zip([xs_logr, gt_xs_logr], ["r", "g"]):
        axes[1].hist(
            torch.clamp(-_logr, min=min_energy, max=max_energy),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            color=color,
            linewidth=4,
        )
    axes[1].set_xlabel("Energy")
    axes[1].legend(["generated data", "test data"])
    axes[1].grid(True)

    return {"visualization/hists": wandb.Image(fig_to_image(fig))}


def visualize(
    energy: BaseEnergy,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
    suffix: str = "",
) -> dict:
    if isinstance(energy, LennardJones):
        if weights is not None:
            warnings.warn(
                "Can't visualize weighted samples for Lennard-Jones energy. Ignoring weights..."
            )
            return {}
        out_dict = viz_lennard_jones(energy, samples)
    else:
        _energy = energy.target_energy if isinstance(energy, IntermediateEnergy) else energy
        if isinstance(_energy, ManyWell):
            out_dict = viz_manywell(energy, samples, weights)
        elif isinstance(_energy, Funnel):
            out_dict = viz_funnel(energy, samples, weights)
        elif isinstance(_energy, (TwentyFiveGaussianMixture, GMM40)):
            out_dict = viz_gmm(energy, samples, weights)
        else:
            warnings.warn(
                f"Warning: {_energy.__class__.__name__} is not supported for visualization."
                + " Skipping..."
            )
            return {}

    plt.close("all")
    return {k.replace("visualization", f"visualization{suffix}"): v for k, v in out_dict.items()}
