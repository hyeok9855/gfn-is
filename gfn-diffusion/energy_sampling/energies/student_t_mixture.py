import numpy as np
import torch
import torch.distributions as D

from energies.base import BaseEnergy
from utils.misc_utils import temp_seed


class StudentTMixture(BaseEnergy):
    def __init__(
        self,
        device: str | torch.device,
        ndim: int = 2,
        num_components: int = 10,
        degree_of_freedom: int = 2,
        ref_gaussian_var: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__(
            device=device,
            ndim=ndim,
            ref_gaussian_var=ref_gaussian_var,
            seed=seed,
            plot_bound=15,
        )

        try:
            locs = torch.from_numpy(np.load(f"energies/data/mos-{ndim}d_locs.npy"))
        except FileNotFoundError:
            with temp_seed(seed):
                locs = (torch.rand(num_components, ndim) * 2 - 1) * 10  # 10 from Beyond ELBOs
        locs = locs.to(dtype=torch.float32, device=device)

        dofs = torch.ones((num_components, ndim), device=device) * degree_of_freedom
        scales = torch.ones((num_components, ndim), device=device)
        logits = torch.ones(num_components, device=device)  # uniform, default in Beyond ELBOs

        mixture_dist = D.Categorical(logits=logits)
        components_dist = D.Independent(
            D.StudentT(loc=locs, scale=scales, df=dofs), reinterpreted_batch_ndims=1  # type: ignore
        )
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
