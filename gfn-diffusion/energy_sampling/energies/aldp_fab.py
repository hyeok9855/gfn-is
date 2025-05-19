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
        energy_cut=1.0e4,  # 1e8 in FAB
        energy_max=1.0e8,  # 1e20 in FAB
        n_threads=4,
        ind_circ_dih=[0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16],
        shift_dih=False,
        shift_dih_params={"hist_bins": 100},
        default_std={"bond": 0.005, "angle": 0.15, "dih": 0.2},
        env="vacuum",
        seed: int = 0,
    ):
        """
        Boltzmann distribution of Alanine dipeptide
        :param data_path: Path to the trajectory file used to initialize the
            transformation, if None, a trajectory is generated
        :type data_path: String
        :param temperature: Temperature of the system
        :type temperature: Integer
        :param energy_cut: Value after which the energy is logarithmically scaled
        :type energy_cut: Float
        :param energy_max: Maximum energy allowed, higher energies are cut
        :type energy_max: Float
        :param n_threads: Number of threads used to evaluate the log
            probability for batches
        :type n_threads: Integer
        :param transform: Which transform to use, can be mixed or internal
        :type transform: String
        """
        super().__init__(device=device, ndim=60, seed=seed)  # 60 is before transform

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
        cart_indices = [8, 6, 14]

        self.ind_circ_dih = ind_circ_dih

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

        self.p = bg.distributions.TransformedBoltzmannParallel(
            system,
            temperature,
            energy_cut=energy_cut,
            energy_max=energy_max,
            transform=self.coordinate_transform,
            n_threads=n_threads,
        )

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
        self.val_xs = torch.tensor(np.concatenate(datas, axis=0), dtype=dtype)
        self.val_log_rs = torch.tensor(np.load(DATA_PATH / "val_log_rs.npy"), dtype=dtype)

    def scale_ind_circ(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, self.ind_circ] = torch.clamp(
            torch.tanh(x[:, self.ind_circ]) * (math.pi + 0.01), min=-math.pi, max=math.pi
        )
        return x

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 60)
        x = self.scale_ind_circ(x)
        x = x.detach().cpu()  # to cpu
        return -self.p.log_prob(x).to(self.device)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                self.energy(copy_x).sum().backward()
                lgv = copy_x.grad
                assert lgv is not None
        return lgv.data

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        with temp_seed(seed or self.seed):
            perm_idx = torch.randperm(self.val_xs.shape[0])[:batch_size]
        return self.val_xs[perm_idx].to(self.device)

    def cached_sample(
        self, batch_size: int, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gt_xs is None or batch_size != self.gt_xs.size(0):
            with temp_seed(seed or self.seed):
                perm_idx = torch.randperm(self.val_xs.shape[0])[:batch_size]
            self.gt_xs = self.val_xs[perm_idx].to(self.device)
            self.gt_xs_log_rewards = self.val_log_rs[perm_idx].to(self.device)
        assert self.gt_xs is not None and self.gt_xs_log_rewards is not None
        return self.gt_xs, self.gt_xs_log_rewards

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 60)
        x = self.scale_ind_circ(x)
        x = x.detach().cpu()  # to cpu
        assert x.shape[1] == 60
        return self.coordinate_transform(x)[0].to(self.device)  # (batch_size, 66)
