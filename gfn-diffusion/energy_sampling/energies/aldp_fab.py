import math
from pathlib import Path

import numpy as np
import torch

import boltzgen as bg
from openmmtools import testsystems

from energies.base import BaseEnergy
from utils.misc_utils import temp_seed


DATA_PATH = Path(__file__).parent / "data" / "aldp_fab"
PI_PLUS_EPS = math.pi + 0.0001


class ALDPFAB(BaseEnergy):
    def __init__(
        self,
        device: str | torch.device = "cpu",
        temperature=300,
        energy_cut=1.0e5,  # 1e8 in FAB
        energy_max=1.0e20,  # 1e20 in FAB
        n_threads=6,
        ind_circ_dih=[0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16],
        shift_dih=False,
        shift_dih_params={"hist_bins": 100},
        default_std={"bond": 0.005, "angle": 0.15, "dih": 0.2},
        env="implicit",
        seed: int = 0,
        chirality_ind=[17, 26],
        chirality_mean_diff=-0.043,
        chirality_threshold=0.8,
        chirality_sharpness=100.0,
    ):
        super().__init__(device=device, ndim=60, seed=seed)  # 60 since we use internal coordinates

        # Define molecule parameters
        z_matrix = [
            (0, [1, 4, 6]),
            (1, [4, 6, 8]),
            (2, [1, 4, 0]),
            (3, [1, 4, 0]),
            (4, [6, 8, 14]),
            (5, [4, 6, 8]),
            (7, [6, 8, 4]),
            (9, [8, 6, 4]),
            (10, [8, 6, 4]),
            (11, [10, 8, 6]),
            (12, [10, 8, 11]),
            (13, [10, 8, 11]),
            (15, [14, 8, 16]),
            (16, [14, 8, 6]),
            (17, [16, 14, 15]),
            (18, [16, 14, 8]),
            (19, [18, 16, 14]),
            (20, [18, 16, 19]),
            (21, [18, 16, 19]),
        ]

        mass = [
            [1.007947, 1.007947, 1.007947],
            [12.01078, 12.01078, 12.01078],
            [1.007947, 1.007947, 1.007947],
            [1.007947, 1.007947, 1.007947],
            [12.01078, 12.01078, 12.01078],
            [15.99943, 15.99943, 15.99943],
            [14.00672, 14.00672, 14.00672],
            [1.007947, 1.007947, 1.007947],
            [12.01078, 12.01078, 12.01078],
            [1.007947, 1.007947, 1.007947],
            [12.01078, 12.01078, 12.01078],
            [1.007947, 1.007947, 1.007947],
            [1.007947, 1.007947, 1.007947],
            [1.007947, 1.007947, 1.007947],
            [12.01078, 12.01078, 12.01078],
            [15.99943, 15.99943, 15.99943],
            [14.00672, 14.00672, 14.00672],
            [1.007947, 1.007947, 1.007947],
            [12.01078, 12.01078, 12.01078],
            [1.007947, 1.007947, 1.007947],
            [1.007947, 1.007947, 1.007947],
            [1.007947, 1.007947, 1.007947],
        ]
        self.mass = torch.tensor(mass, device=self.device).unsqueeze(0)
        self.kBT = 1.380649 * 6.02214076 * 1e-3 * temperature
        self.beta = 1 / self.kBT

        cart_indices = [8, 6, 14]

        # System setup
        if env == "vacuum":
            system = testsystems.AlanineDipeptideVacuum(constraints=None)
        elif env == "implicit":
            system = testsystems.AlanineDipeptideImplicit(constraints=None)
        else:
            raise NotImplementedError("This environment is not implemented.")

        dtype = torch.get_default_dtype()
        transform_data = torch.load(DATA_PATH / "position_min_energy.pt", weights_only=True)
        transform_data = transform_data.to(dtype)

        # Set distribution
        self.coordinate_transform = bg.flows.CoordinateTransform(
            transform_data,
            66,  # 66 is after transform
            z_matrix,
            cart_indices,
            mode="internal",
            ind_circ_dih=ind_circ_dih,
            shift_dih=shift_dih,
            shift_dih_params=shift_dih_params,
            default_std=default_std,
        )
        self.coordinate_transform.to(self.device)

        self.energy_cut = energy_cut

        self.p = bg.distributions.TransformedBoltzmannParallel(
            system,
            temperature,
            energy_cut=energy_cut,
            energy_max=energy_max,
            transform=self.coordinate_transform,
            n_threads=n_threads,
        )
        self.p.to(self.device)

        ncarts = self.coordinate_transform.transform.len_cart_inds
        permute_inv = self.coordinate_transform.transform.permute_inv.cpu().numpy()
        dih_ind = self.coordinate_transform.transform.ic_transform.dih_indices.cpu().numpy()

        ind = torch.arange(self.ndim)
        ind = torch.cat(
            [ind[: 3 * ncarts - 6], -torch.ones(6, dtype=torch.int64), ind[3 * ncarts - 6 :]]
        )
        ind = ind[permute_inv]
        dih_ind = ind[dih_ind]
        self.ind_circ = dih_ind[ind_circ_dih].tolist()

        self.chirality_ind = chirality_ind
        self.chirality_mean_diff = chirality_mean_diff
        self.chirality_threshold = chirality_threshold
        self.chirality_sharpness = chirality_sharpness

        datas = [np.load(DATA_PATH / f"val_before_scale{i}.npy") for i in range(5)]
        approx_sample = torch.tensor(np.concatenate(datas, axis=0), dtype=dtype)
        self.approx_sample = approx_sample[self.get_lform_indices(approx_sample)]

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 60
        x_fab, log_det = self.scale_ind_circ(x)
        energy = -(self.p.log_prob(x_fab) + log_det) + self._compute_chirality_penalty(x)
        energy[energy.isnan()] = 2 * self.energy_cut
        energy[energy > 2 * self.energy_cut] = 2 * self.energy_cut
        return energy

    def get_lform_indices(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the dihedral angle difference
        diff_ = torch.column_stack(
            (
                x[:, self.chirality_ind[0]] - x[:, self.chirality_ind[1]],
                x[:, self.chirality_ind[0]] - x[:, self.chirality_ind[1]] + 2 * np.pi,
                x[:, self.chirality_ind[0]] - x[:, self.chirality_ind[1]] - 2 * np.pi,
            )
        )

        # Find the minimal angular difference (handling periodicity)
        min_diff_ind = torch.min(torch.abs(diff_), dim=1).indices
        diff = diff_[torch.arange(x.shape[0]), min_diff_ind]

        # Compute deviation from the L-form reference
        deviation = torch.abs(diff - self.chirality_mean_diff)

        # High energy for non-L-form
        is_l_form = deviation < self.chirality_threshold
        return is_l_form

    def _compute_chirality_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute smooth chirality penalty that encourages L-form configurations.

        Args:
            x: Input tensor of shape (batch_size, 60) in the internal coordinate space

        Returns:
            Penalty energy for each sample in the batch
        """
        is_l_form = self.get_lform_indices(x)
        penalty = (2 * self.energy_cut) * (1 - is_l_form.float())

        # # Smooth penalty using sigmoid function
        # # - When deviation < threshold: penalty ≈ 0 (L-form region)
        # # - When deviation > threshold: penalty increases smoothly (D-form region)
        # penalty = self.chirality_penalty_strength * torch.sigmoid(
        #     (deviation - self.chirality_threshold) * self.chirality_sharpness
        # )

        return penalty

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        with temp_seed(seed or self.seed):
            perm_idx = torch.randperm(self.approx_sample.shape[0])[:batch_size]
        return self.approx_sample[perm_idx].to(self.device)

    def scale_ind_circ(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[1] == 60
        x_fab = x.clone()
        x_fab[:, self.ind_circ] = torch.tanh(x_fab[:, self.ind_circ]) * PI_PLUS_EPS
        log_cosh_x_ind_circ = torch.log(torch.cosh(x[:, self.ind_circ]))
        log_det = -2 * log_cosh_x_ind_circ + math.log(PI_PLUS_EPS)
        return x_fab, log_det.sum(1)

    def inverse_scale_ind_circ(self, x_fab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert x_fab.shape[1] == 60
        x = x_fab.clone()
        x[:, self.ind_circ] = torch.atanh(x[:, self.ind_circ] / PI_PLUS_EPS)
        log_det = math.log(PI_PLUS_EPS) - torch.log(
            (PI_PLUS_EPS**2 - x_fab[:, self.ind_circ] ** 2).abs()
        )
        return x, log_det.sum(1)

    ##### Some methods for MD #####
    def transform(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # scale_ind_circ --> (60 -> 66)
        assert x.shape[1] == 60
        x_fab, log_det_fab = self.scale_ind_circ(x)
        x_orig, log_det_orig = self.coordinate_transform(x_fab)
        return x_orig, log_det_fab + log_det_orig

    def inverse(self, x_orig: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (66 -> 60) --> inverse_scale_ind_circ
        assert x_orig.shape[1] == 66
        x_fab, log_det_fab = self.coordinate_transform.inverse(x_orig)
        x, log_det_x = self.inverse_scale_ind_circ(x_fab)
        return x, log_det_fab + log_det_x
