from typing import TYPE_CHECKING

import torch
from tqdm import trange

from energies.aldp_fab import ALDPFAB
from mcmcs.base import BaseMCMC

if TYPE_CHECKING:
    from energies import ALDPFAB


class MD(BaseMCMC):
    def __init__(
        self,
        energy: "ALDPFAB",
        gamma: float = 1.0,
        burn_in: int = 50,
        max_iter_ls: int = 100,
        step_size: float = 0.0005,
        **kwargs,
    ) -> None:
        assert isinstance(self.energy, ALDPFAB)
        super().__init__(energy)
        self.gamma = gamma
        self.burn_in = burn_in
        self.step_size = step_size
        self.max_iter_ls = max_iter_ls

        kBT = 1.380649 * 6.02214076 * 1e-3 * self.energy.p.temperature
        self.beta = 1 / kBT

        self.std = torch.sqrt(2 * self.energy.kBT * self.gamma * self.step_size / self.energy.mass)
        self.mol_ndim = self.energy.coordinate_transform.transform.n_dim
        assert self.mol_ndim % 3 == 0
        self.n_atoms = self.mol_ndim // 3

    def sample(self, xs: torch.Tensor):
        assert isinstance(self.energy, ALDPFAB)
        positions = []
        log_rs = []

        x_position = self.energy.transform(xs)

        position = x_position.reshape(-1, self.n_atoms, 3)
        velocity = torch.zeros_like(position, device=position.device)
        position = position.requires_grad_(True)
        energy = self.energy.p.norm_energy(position.reshape(-1, self.mol_ndim)) / self.beta
        force = -torch.autograd.grad(energy.sum(), position)[0]

        for _ in trange(self.max_iter_ls):
            position, velocity, force = self.step(position, velocity, force)
            log_r = -self.energy.p.norm_energy(position.reshape(-1, self.mol_ndim))
            positions.append(position.detach())
            log_rs.append(log_r.detach())

        # stack after burning in first self.burn_in positions and rewards
        positions = torch.stack(positions[self.burn_in :], dim=0)
        log_rs = torch.stack(log_rs[self.burn_in :], dim=0)
        positions = positions.reshape(-1, self.mol_ndim)
        log_rs = log_rs.reshape(-1)
        print(f"{positions.shape[0]} samples collected")

        new_xs, log_det = self.energy.inverse(positions.detach())
        return new_xs, log_rs - log_det

    def step(self, position, velocity, force):
        assert isinstance(self.energy, ALDPFAB)
        with torch.no_grad():
            velocity = (
                (1 - self.gamma * self.step_size) * velocity
                + force * self.step_size / self.energy.mass
                + self.std * torch.randn_like(position, device=position.device)
            )
            position = position + velocity * self.step_size

        position = position.requires_grad_(True)

        energy = self.energy.p.norm_energy(position.reshape(-1, self.mol_ndim)) / self.beta
        force = -torch.autograd.grad(energy.sum(), position)[0]
        return position.detach(), velocity.detach(), force.detach()
