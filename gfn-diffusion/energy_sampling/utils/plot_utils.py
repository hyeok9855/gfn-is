from functools import partial
import itertools
import typing
import warnings
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import mdtraj as md
import seaborn as sns
import torch
import wandb
from matplotlib.colors import LogNorm
from PIL import Image as PILImage

from energies import (
    ALDP,
    ALDPFAB,
    GMM40,
    BaseEnergy,
    Funnel,
    IntermediateEnergy,
    LennardJones,
    ManyWell,
    StudentTMixture,
    TwentyFiveGaussianMixture,
)
from energies.aldp import DATA_PATH as ALDP_DATA_PATH
from utils.particle_system import interatomic_distance


def visualize(
    energy: BaseEnergy,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
    suffix: str = "",
) -> dict:

    _energy_cls = energy.target_energy if isinstance(energy, IntermediateEnergy) else energy
    if isinstance(_energy_cls, ManyWell):
        out_dict = viz_manywell(energy, samples, weights)
    elif isinstance(_energy_cls, Funnel):
        out_dict = viz_funnel(energy, samples, weights)
    elif isinstance(_energy_cls, (TwentyFiveGaussianMixture, GMM40)):
        out_dict = viz_gmm(energy, samples, weights)
    elif isinstance(_energy_cls, StudentTMixture):
        out_dict = viz_student_t_mixture(energy, samples, weights)
    elif isinstance(_energy_cls, LennardJones):
        if weights is not None:
            warnings.warn(
                "Can't visualize weighted samples for Lennard-Jones energy. Ignoring weights..."
            )
            return {}
        out_dict = viz_lennard_jones(energy, samples)
    elif isinstance(_energy_cls, (ALDP, ALDPFAB)):
        if weights is not None:
            warnings.warn("Can't visualize weighted samples for ALDP energy. Ignoring weights...")
            return {}
        out_dict = viz_aldp(energy, samples)
    else:
        warnings.warn(
            f"Warning: {_energy_cls.__class__.__name__} is not supported for visualization."
            + " Skipping..."
        )
        return {}

    plt.close("all")
    return {k.replace("visualization", f"visualization{suffix}"): v for k, v in out_dict.items()}


def sliced_log_reward(x: torch.Tensor, energy: BaseEnergy, dims: tuple) -> torch.Tensor:
    _x = torch.zeros((x.shape[0], energy.ndim))
    _x[:, dims] = x
    return energy.log_reward(_x.to(energy.device), temper=False).detach().cpu()


def fig_to_image(fig):
    fig.canvas.draw()
    return PILImage.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())


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
    log_p_x = log_prob_func(x_points)
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

    fig_contour, ax_contour = viz_contour_sample2d(
        x,
        partial(sliced_log_reward, energy=energy, dims=dims),
        weights=weights,
        lim=lim,
        alpha=alpha,
        n_contour_levels=n_contour_levels,
        grid_width_n_points=grid_width_n_points,
        clamp_min=clamp_min,
    )

    return fig_kde, fig_contour


def viz_energy_hist(energy: BaseEnergy, xs: torch.Tensor) -> dict:
    xs_logr = energy.log_reward(xs, temper=False)
    gt_xs, gt_xs_logr = energy.cached_sample(xs.shape[0])

    xs, gt_xs = xs.cpu(), gt_xs.cpu()
    xs_logr, gt_xs_logr = xs_logr.cpu(), gt_xs_logr.cpu()

    min_energy = (-gt_xs_logr).min()
    max_energy = (-gt_xs_logr).max()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for _logr, color in zip([xs_logr, gt_xs_logr], ["r", "g"]):
        ax.hist(
            torch.clamp(-_logr, min=min_energy, max=max_energy),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            color=color,
            linewidth=4,
        )
    ax.set_xlabel("Energy")
    ax.legend(["generated data", "test data"])
    ax.grid(True)

    return {"visualization/energy_hist": wandb.Image(fig_to_image(fig))}


def viz_interatomic_dist_hist(energy: LennardJones | ALDP | ALDPFAB, xs: torch.Tensor) -> dict:
    gt_xs, _ = energy.cached_sample(xs.shape[0])

    if isinstance(energy, ALDPFAB):
        xs, _ = energy.transform(xs)
        gt_xs, _ = energy.transform(gt_xs)

    xs, gt_xs = xs.cpu(), gt_xs.cpu()
    n_particles = xs.shape[1] // 3

    dist_xs = interatomic_distance(xs, n_particles, 3, True).view(-1)
    dist_gt_xs = interatomic_distance(gt_xs, n_particles, 3, True).view(-1)

    bins = 100
    min_dist = 0
    max_dist = 6

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for _xs, color in zip([dist_xs, dist_gt_xs], ["r", "g"]):
        ax.hist(
            torch.clamp(_xs, min=min_dist, max=max_dist),
            bins=bins,
            alpha=0.5,
            density=True,
            histtype="step",
            color=color,
            linewidth=4,
        )
    ax.set_xlabel("Interatomic Distances")
    ax.legend(["generated data", "test data"])
    ax.grid(True)

    return {"visualization/interatomic_dist_hist": wandb.Image(fig_to_image(fig))}


##### Energy-specific visualization #####


def viz_manywell(
    energy: ManyWell,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> dict:
    lim = energy.plot_bound
    out_dict = {}
    for idx1, idx2 in [(0, 2), (1, 2)]:
        fig_kde, fig_contour = viz_2d_slice(
            energy, (idx1, idx2), samples, weights=weights, lim=lim, kde=False
        )
        out_dict.update(
            {
                f"visualization/kde{idx1}{idx2}": wandb.Image(fig_to_image(fig_kde)),
                f"visualization/contour{idx1}{idx2}": wandb.Image(fig_to_image(fig_contour)),
            }
        )
    out_dict.update(viz_energy_hist(energy, samples))
    return out_dict


def viz_funnel(
    energy: Funnel,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> dict:
    lim = energy.plot_bound
    out_dict = {}
    for i in range(1, 3):
        fig_kde, fig_contour = viz_2d_slice(
            energy, (0, i), samples, weights=weights, lim=lim, kde=False
        )
        out_dict.update(
            {
                f"visualization/kde0{i}": wandb.Image(fig_to_image(fig_kde)),
                f"visualization/contour0{i}": wandb.Image(fig_to_image(fig_contour)),
            }
        )

    return out_dict


def viz_gmm(
    energy: GMM40 | TwentyFiveGaussianMixture,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> dict:
    lim = energy.plot_bound
    out_dict = {}
    for i in range(1, min(energy.ndim, 4), 2):
        fig_kde, fig_contour = viz_2d_slice(
            energy, (i - 1, i), samples, weights=weights, lim=lim, n_contour_levels=100, kde=False
        )
        out_dict.update(
            {
                f"visualization/kde{i - 1}{i}": wandb.Image(fig_to_image(fig_kde)),
                f"visualization/contour{i - 1}{i}": wandb.Image(fig_to_image(fig_contour)),
            }
        )
        plt.close(fig_kde)
        plt.close(fig_contour)

    return out_dict


def viz_student_t_mixture(
    energy: StudentTMixture | IntermediateEnergy,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> dict:
    if energy.ndim != 2:
        samples = samples[:, :2]
        logp_func = partial(sliced_log_reward, energy=energy, dims=(0, 1))
    else:
        logp_func = partial(energy.log_reward, temper=False)

    samples = samples.detach().cpu()
    weights = weights.detach().cpu() if weights is not None else None

    boarder = [-energy.plot_bound, energy.plot_bound]
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    x, y = torch.meshgrid(
        torch.linspace(boarder[0], boarder[1], 100), torch.linspace(boarder[0], boarder[1], 100)
    )
    grid = torch.cat([x.ravel().unsqueeze(1), y.ravel().unsqueeze(1)], dim=1)
    pdf_values = torch.exp(logp_func(grid))
    pdf_values = pdf_values.reshape(x.shape)
    ax.contourf(x, y, pdf_values, levels=50)

    if samples is not None:
        samples = samples.detach().cpu()
        weights = weights.detach().cpu() if weights is not None else None
        plt.scatter(
            samples[:, 0],
            samples[:, 1],
            c="r",
            alpha=0.5,
            marker="x",
            s=weights[: len(samples)] * len(samples) * 5 if weights is not None else 5,
        )
    ax.set_xlim(boarder[0], boarder[1])
    ax.set_ylim(boarder[0], boarder[1])
    return {"visualization/contour": wandb.Image(fig_to_image(fig))}


def viz_lennard_jones(energy: LennardJones, xs: torch.Tensor) -> dict:
    out_dict = {}
    out_dict.update(viz_interatomic_dist_hist(energy, xs))
    out_dict.update(viz_energy_hist(energy, xs))
    return out_dict


def viz_aldp(energy: ALDP | ALDPFAB, xs: torch.Tensor) -> dict:
    out_dict = {}
    out_dict.update(plot_phi_psi(energy, xs))
    out_dict.update(draw_mols(energy, xs))
    out_dict.update(viz_energy_hist(energy, xs))
    out_dict.update(viz_interatomic_dist_hist(energy, xs))
    return out_dict


##### ALDP #####

ATOM_COLORS = {
    "carbon": "gray",
    "nitrogen": "blue",
    "oxygen": "red",
    "hydrogen": "black",
    "sulfur": "yellow",
    "phosphorus": "orange",
}

ATOM_SIZES = {
    "carbon": 100,
    "nitrogen": 100,
    "oxygen": 100,
    "hydrogen": 25,
    "sulfur": 100,
    "phosphorus": 100,
}


def plot_phi_psi(energy: ALDP | ALDPFAB, xs: torch.Tensor, dpi=300):
    """
    Plots a 2D histogram of phi and psi angles.

    Args:
        xs (torch.Tensor): Input data for dihedral angle computation.
        dpi (int): Dots per inch for the figure.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """

    if isinstance(energy, ALDPFAB):
        xs, _ = energy.transform(xs)

    assert xs.ndim == 2  # (n_samples, n_atoms * 3)
    xs = xs.reshape(xs.shape[0], -1, 3)  # (n_samples, n_atoms, 3)

    def compute_dihedral(positions: torch.Tensor) -> torch.Tensor:
        v = positions[:, :-1] - positions[:, 1:]
        v0 = -v[:, 0]
        v1 = v[:, 2]
        v2 = v[:, 1]

        s0 = torch.sum(v0 * v2, dim=-1, keepdim=True) / torch.sum(v2 * v2, dim=-1, keepdim=True)
        s1 = torch.sum(v1 * v2, dim=-1, keepdim=True) / torch.sum(v2 * v2, dim=-1, keepdim=True)

        v0 = v0 - s0 * v2
        v1 = v1 - s1 * v2

        v0 = v0 / torch.norm(v0, dim=-1, keepdim=True)
        v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
        v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)

        x = torch.sum(v0 * v1, dim=-1)
        v3 = torch.cross(v0, v2, dim=-1)
        y = torch.sum(v3 * v1, dim=-1)
        return torch.atan2(y, x)

    fig = plt.figure(figsize=(7, 6), dpi=dpi)

    angle_1 = [6, 8, 14, 16]
    angle_2 = [1, 6, 8, 14]

    psi = compute_dihedral(xs[:, angle_1, :])
    phi = compute_dihedral(xs[:, angle_2, :])
    phi = phi.detach().cpu().numpy()
    psi = psi.detach().cpu().numpy()

    xedges = np.linspace(-np.pi, np.pi, 51)
    yedges = np.linspace(-np.pi, np.pi, 51)
    plt.hist2d(phi, psi, bins=[xedges, yedges], norm=LogNorm(), cmap="viridis")
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel("$\phi$", fontsize=14)
    plt.ylabel("$\psi$", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.colorbar(label="Density")
    plt.tight_layout()
    return {"visualization/phi_psi": wandb.Image(fig_to_image(fig))}


def draw_mols(energy: ALDP | ALDPFAB, xs: torch.Tensor, name="aldp"):
    """
    Draw a figure containing 3D molecule.

    Args:
        energy (BaseEnergy): Energy function.
        sample (Array): Sample generated by model.

    Return:
        fig, axs: matplotlib figure and axes objec
    """

    if isinstance(energy, ALDPFAB):
        xs, _ = energy.transform(xs)

    assert xs.shape[0] >= 3

    # Make ten subplots
    fig, axs = plt.subplots(1, 3, figsize=(30, 10), subplot_kw=dict(projection="3d"))

    for i, ax in enumerate(axs.flatten()):
        draw_mol(
            name,
            ax,
            xs[i].reshape(-1, 3).detach().cpu().numpy(),
        )
    return {"visualization/3D": wandb.Image(fig_to_image(fig))}


@typing.no_type_check
def draw_mol(name: str, ax: plt.Axes, coordinate: np.ndarray) -> plt.Axes:
    """
    Visualizes molecular conformation using matplotlib's 3D plot.
    Returns the generated matplotlib Axes object.

    parameters:
        coordinates (Array): Molecular atom coordinates. Should be array of shape (n_atoms, 3).

    return:
        matplotlib.axes.Axes: Axes object containing the visualized molecular plot.
    """

    # get topology (md.Topology) from pdb file
    coordinate = np.nan_to_num(coordinate, nan=0.0, posinf=0.0, neginf=0.0)

    topology = md.load(ALDP_DATA_PATH / f"{name}.pdb").topology

    center_of_mass = np.mean(coordinate, axis=0)
    coordinate = coordinate - center_of_mass

    # Set the box aspect ratio
    ax.set_aspect("equal")

    # Set the background color to white
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")

    # Draw atoms
    for i, atom in enumerate(topology.atoms):
        atom_name = atom.element.name
        ax.scatter(
            coordinate[i, 0],
            coordinate[i, 1],
            coordinate[i, 2],
            c=ATOM_COLORS.get(atom_name, "gray"),
            s=ATOM_SIZES.get(atom_name, 20),
            label=atom_name,
            alpha=0.8,
            edgecolors="black",
            depthshade=True,
        )

    # Draw bonds
    for bond in topology.bonds:
        atom1, atom2 = bond
        x = [coordinate[atom1.index, 0], coordinate[atom2.index, 0]]
        y = [coordinate[atom1.index, 1], coordinate[atom2.index, 1]]
        z = [coordinate[atom1.index, 2], coordinate[atom2.index, 2]]
        ax.plot(x, y, z, "k-", linewidth=2.0, alpha=0.6)

    # Set the view angle
    ax.view_init(elev=20, azim=45)

    # Set the axis labels
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")

    # Adjust the axis limits to fit the molecule
    max_range = (
        np.array(
            [
                coordinate[:, 0].max() - coordinate[:, 0].min(),
                coordinate[:, 1].max() - coordinate[:, 1].min(),
                coordinate[:, 2].max() - coordinate[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
    mid_x = (coordinate[:, 0].max() + coordinate[:, 0].min()) * 0.5
    mid_y = (coordinate[:, 1].max() + coordinate[:, 1].min()) * 0.5
    mid_z = (coordinate[:, 2].max() + coordinate[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Draw the legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    return ax
