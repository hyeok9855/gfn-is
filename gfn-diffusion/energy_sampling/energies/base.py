import abc

import torch


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
