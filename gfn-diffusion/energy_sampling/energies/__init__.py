import torch
from .base import BaseEnergy
from .funnel import Funnel
from .gmm40 import GMM40
from .lgcp import LGCP
from .manywell import ManyWell
from .twenty_five_gmm import TwentyFiveGaussianMixture


def get_energy(energy_name: str, ndim: int, device: torch.device) -> BaseEnergy:
    if energy_name == "25gmm":
        if ndim != 2:
            raise ValueError("25GMM is only supported for 2D")
        energy = TwentyFiveGaussianMixture(device=device)
    elif energy_name == "gmm40":
        energy = GMM40(device=device, ndim=ndim)
    elif energy_name == "funnel":
        energy = Funnel(device=device, ndim=ndim)
    elif energy_name == "many_well":
        energy = ManyWell(device=device, ndim=ndim)
    elif energy_name == "lgcp":
        raise NotImplementedError
        # TODO: ndim?
        energy = LGCP(device=device)
    else:
        raise ValueError(f"Unknown energy: {energy_name}")
    return energy
