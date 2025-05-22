from pathlib import Path
import torch
import torchani
import numpy as np
from ase.io import read
from ase.data import chemical_symbols
from torchani.units import hartree2kjoulemol

from energies.base import BaseEnergy
from utils.misc_utils import temp_seed
from utils.particle_system import remove_mean


DATA_PATH = Path(__file__).parent / "data" / "aldp"


class ALDP(BaseEnergy):
    def __init__(self, device: str | torch.device, seed: int = 0):

        molecule = read(DATA_PATH / "aldp.pdb")
        atomic_numbers = molecule.get_atomic_numbers()  # type: ignore

        self.n_particles = len(atomic_numbers)
        self.spatial_dim = 3

        super().__init__(device=device, ndim=self.n_particles * self.spatial_dim, seed=seed)

        self.model = torchani.models.ANI2x().to(self.device)
        atomic_symbols = [chemical_symbols[z] for z in atomic_numbers]
        self.species = self.model.consts.species_to_tensor(atomic_symbols).unsqueeze(0)

        target_temperature = 300
        kBT = 1.380649 * 6.02214076 * 1e-3 * target_temperature
        self.beta = 1 / kBT

        atom_types = torch.arange(self.n_particles).to(self.device)
        atom_types[[0, 2, 3]] = 2
        atom_types[[11, 12, 13]] = 12
        atom_types[[19, 20, 21]] = 20
        self.h_initial = torch.nn.functional.one_hot(atom_types, num_classes=22).float()

        datas = [np.load(DATA_PATH / f"aldp{i}.npy") for i in range(6)]
        data = torch.tensor(np.concatenate(datas, axis=0))
        self.approx_sample = remove_mean(data, self.n_particles, self.spatial_dim)

    def energy(self, x: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        x = x.reshape(-1, self.n_particles, self.spatial_dim)
        species = self.species.repeat(x.shape[0], 1)
        energies_h = self.model((species, 10 * x)).energies
        energies_kJmol = hartree2kjoulemol(energies_h)
        if reduce:
            return energies_kJmol * self.beta
        else:
            return energies_kJmol

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        with temp_seed(seed or self.seed):
            perm_idx = torch.randperm(self.approx_sample.shape[0])[:batch_size]
        return self.approx_sample[perm_idx].to(self.device)
