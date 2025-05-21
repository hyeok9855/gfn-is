import math
from pathlib import Path

import numpy as np
import torch

import boltzgen as bg
from openmmtools import testsystems

from energies.base import BaseEnergy
from utils.misc_utils import temp_seed


DATA_PATH = Path(__file__).parent / "data" / "aldp_fab"


class ALDPFAB(BaseEnergy):
    def __init__(
        self,
        device: str | torch.device = "cpu",
        temperature=300,
        energy_cut=1.0e6,  # 1e8 in FAB
        energy_max=1.0e20,  # 1e20 in FAB
        n_threads=6,
        ind_circ_dih=[0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16],
        shift_dih=False,
        shift_dih_params={"hist_bins": 100},
        default_std={"bond": 0.005, "angle": 0.15, "dih": 0.2},
        env="implicit",
        ref_gaussian_var: float = 1.0,
        seed: int = 0,
    ):
        super().__init__(
            device=device,
            ndim=60,
            ref_gaussian_var=ref_gaussian_var,
            seed=seed,
        )  # 60 is before transform

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

        datas = [np.load(DATA_PATH / f"val_before_scale{i}.npy") for i in range(5)]
        self.approx_sample = torch.tensor(np.concatenate(datas, axis=0), dtype=dtype)

    def energy_in_original_space(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 66
        return self.p.norm_energy(x)

    def scale_ind_circ(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 60
        copy_x = x.clone()
        copy_x[:, self.ind_circ] = torch.clamp(
            torch.tanh(copy_x[:, self.ind_circ]) * (math.pi + 0.01), min=-math.pi, max=math.pi
        )
        return copy_x

    def inverse_scale_ind_circ(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 60
        copy_x = x.clone()
        copy_x[:, self.ind_circ] = torch.atanh(copy_x[:, self.ind_circ] / (math.pi + 0.01))
        return copy_x

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 60
        x = self.scale_ind_circ(x)
        return -self.p.log_prob(x)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # scale_ind_circ --> (60 -> 66)
        assert x.shape[1] == 60
        x = self.scale_ind_circ(x)
        return self.coordinate_transform(x)[0]  # (batch_size, 66)

    def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (66 -> 60) --> inverse_scale_ind_circ
        assert x.shape[1] == 66
        x, log_det = self.coordinate_transform.inverse(x)
        x = self.inverse_scale_ind_circ(x)
        return x, log_det

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        with temp_seed(seed or self.seed):
            perm_idx = torch.randperm(self.approx_sample.shape[0])[:batch_size]
        return self.approx_sample[perm_idx].to(self.device)
