import math
from typing import Literal

import torch
import torch.nn as nn

from energies import BaseEnergy
from models.modules import BaseModule

logtwopi = math.log(2 * math.pi)


def cos_sq_fn_step_scheme(n_steps, s=0.008, noise_scale=6.0, dtype=torch.float32):
    pre_phase = torch.linspace(0, 1, n_steps + 1, dtype=dtype)
    phase = ((pre_phase + s) / (1 + s)) * torch.pi * 0.5
    dts = torch.cos(phase) ** 4
    dts_out = dts / dts.sum()
    return dts_out * noise_scale


class GFN(nn.Module):
    def __init__(
        self,
        energy: BaseEnergy,
        module: BaseModule,
        device=torch.device("cuda"),
        num_steps: int | None = None,
        reference_process: Literal["pinned_brownian", "ou"] = "pinned_brownian",
        # --- Pinned Brownian Args --- #
        t_scale: float | None = None,
        # --- OU Args --- #
        init_std: float | None = None,
        noise_scale: float | None = None,
        # --- SubTB Args --- #
        partial_energy: bool = False,
        learn_beta: bool = False,
    ) -> None:
        super().__init__()
        self.energy = energy
        self.pred_module = module
        self.device = device

        self.num_steps = num_steps
        self.dt = torch.tensor(1.0 / num_steps, device=self.device)

        self.reference_process = reference_process
        self.t_scale = self.init_std = self.noise_scale = None

        if reference_process == "pinned_brownian":
            assert t_scale is not None
            self.t_scale = t_scale
            self.sample_initial_state = lambda bsz: torch.zeros(
                (bsz, self.energy.ndim), device=self.device
            )
            self.initial_logprob = lambda s: torch.zeros((s.shape[0],), device=self.device)
            self.alpha_fn = self.lambda_fn = None
        elif reference_process == "ou":
            assert init_std is not None and noise_scale is not None
            self.init_std = init_std
            self.noise_scale = noise_scale
            self.initial_dist = torch.distributions.Normal(
                torch.zeros((self.energy.ndim,), device=self.device),
                torch.full((self.energy.ndim,), init_std, device=self.device),
            )
            self.sample_initial_state = lambda bsz: self.initial_dist.sample(
                sample_shape=torch.Size((bsz,))
            )
            self.initial_logprob = lambda s: self.initial_dist.log_prob(s)
            alphas = cos_sq_fn_step_scheme(num_steps, noise_scale=noise_scale)
            self.alpha_fn = lambda step: alphas[step]
            self.lambda_fn = lambda step: alphas[step]
        else:
            raise ValueError(f"Invalid reference process: {reference_process}")

        self.partial_energy = partial_energy
        self.beta_model = None
        if learn_beta:
            self.beta_model = torch.nn.Parameter(
                torch.cat([torch.ones(self.num_steps, device=self.device)])
            )
            self.softplus = nn.Softplus()

    def forward_step(
        self,
        s: torch.Tensor,  # state at time t
        s_next: torch.Tensor | None,  # state at time t + \Delta t; if None, we sample
        step: int,  # step at time t; not used here
        pf_mean: torch.Tensor,
        pf_logvar_correction: torch.Tensor,
        detach: bool = True,  # for PIS
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.reference_process == "pinned_brownian":
            fwd_mean = s + self.dt * pf_mean
            fwd_std = (self.dt * math.sqrt(self.t_scale)).sqrt() * (pf_logvar_correction / 2).exp()
        elif self.reference_process == "ou":
            sqrt_at = torch.clamp(
                torch.tensor(self.alpha_fn(step), device=s.device).sqrt(), 0.0, 1.0
            )
            sqrt_1_minus_at = (1 - sqrt_at**2).sqrt()
            fwd_mean = sqrt_1_minus_at * s + sqrt_at**2 * pf_mean
            fwd_std = sqrt_at * self.init_std * (pf_logvar_correction / 2).exp()
        else:
            raise ValueError(f"Invalid reference process: {self.reference_process}")

        if s_next is None:
            s_next = fwd_mean + fwd_std * torch.randn_like(s, device=s.device)
            s_next = s_next.detach() if detach else s_next

        noise = (s_next - fwd_mean) / fwd_std
        log_pfs = -0.5 * (logtwopi + 2 * fwd_std.log() + noise**2).sum(1)
        return s_next, log_pfs

    def backward_step(
        self,
        s: torch.Tensor | None,  # state at time t; if None, we sample
        s_next: torch.Tensor,  # state at time t + \Delta t
        step: int,  # step at time t
        pb_mean_correction: torch.Tensor,
        pb_var_correction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.reference_process == "pinned_brownian":
            t_next = (step + 1) / self.num_steps

            bwd_mean = s_next - s_next * self.dt / t_next * pb_mean_correction
            bwd_var = self.t_scale * self.dt * (t_next - self.dt) / t_next * pb_var_correction
            bwd_std = bwd_var.sqrt()
        elif self.reference_process == "ou":
            sqrt_at = torch.clamp(
                torch.tensor(self.alpha_fn(step), device=s.device).sqrt(), 0.0, 1.0
            )
            sqrt_1_minus_at = (1 - sqrt_at**2).sqrt()
            bwd_mean = sqrt_1_minus_at * s_next
            bwd_std = sqrt_at * self.init_std * (pb_var_correction / 2).exp()
        else:
            raise ValueError(f"Invalid reference process: {self.reference_process}")

        if self.reference_process == "pinned_brownian" and step == 0:
            s = torch.zeros_like(s_next)
            log_pbs = torch.zeros_like(bwd_mean[:, 0])
        else:
            if s is None:
                s = bwd_mean + bwd_std * torch.randn_like(s_next)
                s = s.detach()

            noise = (s - bwd_mean) / bwd_std
            log_pbs = -0.5 * (logtwopi + 2 * bwd_std.log() + noise**2).sum(1)
        return s, log_pbs

    def get_partial_energy(
        self,
        states: torch.Tensor,  # (bsz, T', ndim)
        steps: torch.Tensor,  # (T')
    ) -> torch.Tensor:
        assert self.partial_energy
        bsz = states.shape[0]

        ts = steps / self.num_steps

        if self.beta_model is not None:
            betas = self.softplus(self.beta_model).cumsum(0)
            betas = betas / betas[-1]
            betas = torch.cat([torch.zeros(1), betas], dim=0)
            betas = betas[steps]
        else:
            betas = ts

        ref_log_var = (self.t_scale * ts).log().unsqueeze(2)  # (bsz, T', 1)
        log_p_ref = -0.5 * (logtwopi + ref_log_var + (-ref_log_var).exp() * (states**2)).sum(-1)
        # (bsz, T')
        partial_energy = (1 - betas) * log_p_ref + betas * self.energy.log_reward(
            states.reshape(-1, self.energy.ndim)
        ).view(bsz, -1)
        return partial_energy  # (bsz, T')

    def get_trajectory_fwd(
        self,
        batch_size: int,
        pis=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = batch_size
        s = self.sample_initial_state(bsz)
        init_log_probs = self.initial_logprob(s)

        log_pfs = torch.zeros((bsz, self.num_steps), device=self.device)
        log_pbs = torch.zeros((bsz, self.num_steps), device=self.device)
        log_fs = torch.zeros((bsz, self.num_steps + 1), device=self.device)
        states = torch.zeros((bsz, self.num_steps + 1, self.energy.ndim), device=self.device)
        states[:, 0] = s

        for i in range(self.num_steps):  # from step 0 to self.num_steps - 1
            pf_mean, pf_logvar, flow = self.pred_module.forward(
                s, self.dt * i, self.energy.grad_log_reward
            )

            if self.pred_module.conditional_flow_model:
                log_fs[:, i] = flow

            s_, log_pfs[:, i] = self.forward_step(s, None, i, pf_mean, pf_logvar, detach=not pis)

            mean_correction, var_correction = self.pred_module.backward(s_, self.dt * (i + 1))
            _, log_pbs[:, i] = self.backward_step(s, s_, i, mean_correction, var_correction)

            s = s_
            states[:, i + 1] = s

        # Assign the terminal reward
        # Set terminal reward based on whether we need gradients for PIS loss
        with torch.enable_grad() if pis else torch.no_grad():
            log_fs[:, -1] = self.energy.log_reward(states[:, -1])

        if self.partial_energy:
            log_fs[:, 1:-1] += self.get_partial_energy(
                states[:, 1:-1], torch.arange(1, self.num_steps, device=self.device)
            )

        return states, log_pfs, log_pbs, log_fs, init_log_probs

    def get_trajectory_bwd(
        self,
        s: torch.Tensor,
        log_r: torch.Tensor,  # (bsz,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = s.shape[0]

        log_pfs = torch.zeros((bsz, self.num_steps), device=self.device)
        log_pbs = torch.zeros((bsz, self.num_steps), device=self.device)
        log_fs = torch.zeros((bsz, self.num_steps + 1), device=self.device)
        states = torch.zeros((bsz, self.num_steps + 1, self.energy.ndim), device=self.device)
        states[:, -1] = s

        for i in range(self.num_steps - 1, -1, -1):  # from step T - 1 to 0
            mean_correction, var_correction = self.pred_module.backward(s, self.dt * (i + 1))
            s_, log_pbs[:, i] = self.backward_step(None, s, i, mean_correction, var_correction)

            pf_mean, pf_logvar, flow = self.pred_module.forward(
                s_, self.dt * i, self.energy.grad_log_reward
            )

            if self.pred_module.conditional_flow_model:
                log_fs[:, i] = flow

            _, log_pfs[:, i] = self.forward_step(s_, s, i, pf_mean, pf_logvar, detach=True)

            s = s_
            states[:, i] = s

        log_fs[:, -1] = log_r

        if self.partial_energy:
            log_fs[:, 1:-1] += self.get_partial_energy(
                states[:, 1:-1], torch.arange(1, self.num_steps, device=self.device)
            )

        init_log_probs = self.initial_logprob(s)

        return states, log_pfs, log_pbs, log_fs, init_log_probs

    def get_trajectory_fwd_smc(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError  # TODO: implement SMC
        bsz = batch_size
        s = self.sample_initial_state(batch_size)
        init_log_probs = self.initial_logprob(s)

        log_pfs = torch.zeros((bsz, self.num_steps), device=self.device)
        log_pbs = torch.zeros((bsz, self.num_steps), device=self.device)
        log_fs = torch.zeros((bsz, self.num_steps + 1), device=self.device)
        states = torch.zeros((bsz, self.num_steps + 1, self.energy.ndim), device=self.device)
        states[:, 0] = s

        for i in range(self.num_steps):  # from step 0 to self.num_steps - 1
            pf_mean, pf_logvar, flow = self.pred_module.forward(
                s, self.dt * i, self.energy.grad_log_reward
            )

            if self.pred_module.conditional_flow_model:
                log_fs[:, i] = flow

            s_, log_pfs[:, i] = self.forward_step(s, None, i, pf_mean, pf_logvar, detach=not pis)

            mean_correction, var_correction = self.pred_module.backward(s_, self.dt * (i + 1))
            _, log_pbs[:, i] = self.backward_step(s, s_, i, mean_correction, var_correction)

            s = s_
            states[:, i + 1] = s

        # Assign the terminal reward
        # Set terminal reward based on whether we need gradients for PIS loss
        with torch.no_grad():
            log_fs[:, -1] = self.energy.log_reward(states[:, -1])

        if self.partial_energy:
            log_fs[:, 1:-1] += self.get_partial_energy(
                states[:, 1:-1], torch.arange(1, self.num_steps, device=self.device)
            )

        return states, log_rs, log_iws
