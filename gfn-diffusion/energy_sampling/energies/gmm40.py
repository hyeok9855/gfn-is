import torch
import torch.distributions as D

from energies.base import BaseEnergy
from utils.misc_utils import temp_seed


class GMM40(BaseEnergy):
    def __init__(
        self,
        device: str | torch.device,
        ndim: int = 2,
        num_components: int = 40,
        loc_scaling: float = 4.0,
        scale_scaling: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__(device=device, ndim=ndim, plot_bound=loc_scaling * 1.5)
        self.device = device

        self.seed = seed

        with temp_seed(seed):
            logits = torch.ones(num_components, device=device)
            mean = (torch.rand(num_components, ndim, device=device) * 2 - 1) * loc_scaling
            scale = torch.ones(num_components, ndim, device=device) * scale_scaling

        mixture_dist = D.Categorical(logits=logits)
        components_dist = D.Independent(D.Normal(loc=mean, scale=scale), reinterpreted_batch_ndims=1)
        self.distribution = D.MixtureSameFamily(mixture_dist, components_dist)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return -self._log_prob(x)

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        with temp_seed(seed or self.seed):
            return self.distribution.sample(sample_shape=torch.Size((batch_size,))).to(self.device)

    def gt_logz(self):
        return 0.0

    # ----- Energy-specific methods ----- #
    def _log_prob(self, x: torch.Tensor) -> torch.Tensor:
        batched = x.ndim == 2
        if not batched:
            x = x.unsqueeze(0)

        log_prob = self.distribution.log_prob(x)

        if not batched:
            log_prob = log_prob.squeeze(0)

        return log_prob
