import abc

import torch

from models.gfn import GFN


class BaseEnergy(abc.ABC):
    def __init__(self, device: str | torch.device, ndim: int, plot_bound: float = 1.0) -> None:
        self.device = device
        self.ndim = ndim
        self.plot_bound = plot_bound

    @abc.abstractmethod
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_reward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.energy(x)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                self.energy(copy_x).sum().backward()
                lgv = copy_x.grad
                assert lgv is not None
        return lgv.data

    def sample(self, batch_size: int) -> torch.Tensor:
        raise NotImplementedError

    def gt_logz(self) -> float:
        raise NotImplementedError


class IntermediateEnergy(BaseEnergy):
    def __init__(self, target_energy: BaseEnergy, gfn: GFN, t: float) -> None:
        super().__init__(target_energy.device, target_energy.ndim, target_energy.plot_bound)
        self.target_energy = target_energy
        self.gfn = gfn
        self.t = t

    def energy(self, states: torch.Tensor) -> torch.Tensor:
        return -self.log_reward(states)

    def log_reward(self, states: torch.Tensor) -> torch.Tensor:
        # states: (bsz, ndim)

        if self.t == 1.0:
            log_fs = self.target_energy.log_reward(states)
            return log_fs

        ts = torch.ones_like(states[:, 0]) * self.t
        with torch.no_grad():
            _, log_fs = self.gfn.module.predict_forward(states, ts, self.target_energy.log_reward)
            if self.gfn.partial_energy:
                log_fs += self.gfn.get_partial_energy(
                    states.unsqueeze(1), ts.unsqueeze(1), self.target_energy.log_reward
                ).squeeze(1)
        return log_fs
