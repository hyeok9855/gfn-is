import argparse

import torch

from .base import BaseEnergy
from .funnel import Funnel
from .gmm40 import GMM40
from .lennard_jones import LJ13, LJ55, LennardJones
from .lgcp import LGCP
from .manywell import ManyWell
from .twenty_five_gmm import TwentyFiveGaussianMixture

from .intermediate_energy import IntermediateEnergy


def get_energy(args: argparse.Namespace, device: torch.device) -> BaseEnergy:
    energy_name: str = args.energy_name
    ndim: int = args.ndim

    if energy_name == "25gmm":
        if ndim != 2:
            raise ValueError("25GMM is only supported for 2D")
        energy = TwentyFiveGaussianMixture(device=device)
    elif energy_name == "gmm40":
        energy = GMM40(device=device, ndim=ndim)
    elif energy_name == "funnel":
        energy = Funnel(device=device, ndim=ndim)
    elif energy_name == "manywell":
        energy = ManyWell(device=device, ndim=ndim)
    elif energy_name == "lgcp":
        energy = LGCP(device=device)
    elif energy_name == "lj13":
        energy = LJ13(device=device)
    elif energy_name == "lj55":
        energy = LJ55(device=device)
    else:
        raise ValueError(f"Unknown energy: {energy_name}")
    return energy
