import torch

from energies.base import BaseEnergy
from models.gfn import GFN


class IntermediateEnergy(BaseEnergy):
    def __init__(self, target_energy: BaseEnergy, gfn: GFN, t: float) -> None:
        super().__init__(
            device=target_energy.device,
            ndim=target_energy.ndim,
            ref_gaussian_var=target_energy.ref_gaussian_var,
            seed=target_energy.seed,
            plot_bound=target_energy.plot_bound,
        )
        self.target_energy = target_energy
        self.gfn = gfn
        self.t = t

    def energy(self, states: torch.Tensor) -> torch.Tensor:
        return -self.log_reward(states)

    def log_reward(self, states: torch.Tensor, temper: bool = False) -> torch.Tensor:
        # states: (bsz, ndim)

        if self.t == 1.0:
            log_fs = self.target_energy.log_reward(states, temper=temper)
            return log_fs

        ts = torch.ones_like(states[:, 0]) * self.t
        with torch.no_grad():
            _, _, log_fs = self.gfn.pred_module.predict_forward(
                states, ts, self.target_energy.log_reward
            )
            if self.gfn.partial_energy:
                log_fs += self.gfn.get_partial_energy(states.unsqueeze(1), ts.unsqueeze(1)).squeeze(
                    1
                )
        return log_fs
