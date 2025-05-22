import abc

import torch
import torch.distributions as D


class BaseEnergy(abc.ABC):
    def __init__(
        self,
        device: str | torch.device,
        ndim: int,
        seed: int = 0,
        plot_bound: float = 1.0,
    ) -> None:
        self.device = device
        self.ndim = ndim
        self.seed = seed
        self.plot_bound = plot_bound
        self.gt_xs: torch.Tensor | None = None
        self.gt_xs_log_rewards: torch.Tensor | None = None
        self._invtemp = 1.0

    @property
    def invtemp(self) -> float:
        return self._invtemp

    @invtemp.setter
    def invtemp(self, value: float) -> None:
        self._invtemp = value

    @abc.abstractmethod
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_reward(self, x: torch.Tensor, temper: bool = True) -> torch.Tensor:
        log_r = -self.energy(x)
        if temper:
            return self.invtemp * log_r
        return log_r

    def grad_log_reward(self, x: torch.Tensor, temper: bool = True) -> torch.Tensor:
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                self.log_reward(copy_x, temper=temper).sum().backward()
                lgv = copy_x.grad
                assert lgv is not None
        return lgv.data

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        raise NotImplementedError

    def gt_logz(self) -> float:
        raise NotImplementedError

    def cached_sample(
        self, batch_size: int, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gt_xs is None or batch_size != self.gt_xs.size(0):
            self.gt_xs = self.sample(batch_size, seed)
            self.gt_xs_log_rewards = self.log_reward(self.gt_xs, temper=False)
        assert self.gt_xs_log_rewards is not None
        return self.gt_xs, self.gt_xs_log_rewards
