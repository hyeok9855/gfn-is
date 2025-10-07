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
        pf_logvar: torch.Tensor,
        detach: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.reference_process == "pinned_brownian":
            return forward_step_pinned_brownian(
                s, s_next, step, self.num_steps, pf_mean, pf_logvar, self.t_scale, detach
            )
        else:  # TODO
            raise ValueError(f"Invalid reference process: {self.reference_process}")

    def backward_step(
        self,
        s: torch.Tensor | None,  # state at time t; if None, we sample
        s_next: torch.Tensor,  # state at time t + \Delta t
        step: int,  # step at time t
        pb_mean_correction: torch.Tensor,
        pb_var_correction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.reference_process == "pinned_brownian":
            return backward_step_pinned_brownian(
                s, s_next, step, self.num_steps, pb_mean_correction, pb_var_correction, self.t_scale
            )
        else:  # TODO
            raise ValueError(f"Invalid reference process: {self.reference_process}")

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


def forward_step_pinned_brownian(
    s: torch.Tensor,  # state at time t
    s_next: torch.Tensor | None,  # state at time t + \Delta t; if None, we sample
    step: int,  # step at time t; not used here
    num_steps: int,
    pf_mean: torch.Tensor,
    pf_logvar: torch.Tensor,
    t_scale: float,
    detach: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    # s.shape = (bsz, ndim)
    # s_next.shape = (bsz, ndim)
    # pf_mean.shape = (bsz, ndim)
    # pf_logvar.shape = (bsz, ndim)

    dt = torch.tensor(1.0 / num_steps, device=s.device)

    pf_logvar = pf_logvar + math.log(t_scale)

    # PIS requires gradients w.r.t. the parameters
    if detach:
        pf_mean_sample = pf_mean.detach()
        pf_logvar_sample = pf_logvar.detach()
    else:
        pf_mean_sample = pf_mean
        pf_logvar_sample = pf_logvar

    if s_next is None:
        s_next = (
            s
            + dt * pf_mean_sample
            + dt.sqrt() * (pf_logvar_sample / 2).exp() * torch.randn_like(s, device=s.device)
        )

    noise = ((s_next - s) - dt * pf_mean) / (dt.sqrt() * (pf_logvar / 2).exp())
    log_pfs = -0.5 * (noise**2 + logtwopi + dt.log() + pf_logvar).sum(1)

    return s_next, log_pfs


def backward_step_pinned_brownian(
    s: torch.Tensor | None,  # state at time t; if None, we sample
    s_next: torch.Tensor,  # state at time t + \Delta t
    step: int,  # step at time t
    num_steps: int,
    pb_mean_correction: torch.Tensor,
    pb_var_correction: torch.Tensor,
    t_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # s.shape = (bsz, ndim)
    # s_next.shape = (bsz, ndim)
    # pb_mean_correction.shape = (bsz, ndim)
    # pb_var_correction.shape = (bsz, ndim)

    dt = torch.tensor(1.0 / num_steps, device=s_next.device)
    t_next = (step + 1) / num_steps

    back_mean = s_next - s_next * dt / t_next * pb_mean_correction
    back_var = t_scale * dt * (t_next - dt) / t_next * pb_var_correction

    if s is None:
        if step == 0:
            s = torch.zeros_like(s_next)
        else:
            s = back_mean.detach() + back_var.sqrt().detach() * torch.randn_like(s_next)

    noise_backward = (s - back_mean) / back_var.sqrt()
    if step == 0:
        log_pbs = torch.zeros_like(back_mean[:, 0])
    else:
        log_pbs = -0.5 * (noise_backward**2 + logtwopi + back_var.log()).sum(1)
    return s, log_pbs


def forward_step_ou(
    s: torch.Tensor,  # state at time t
    s_next: torch.Tensor | None,  # state at time t + \Delta t; if None, we sample
    step: int,  # step at time t; not used here
    num_steps: int,
    pf_mean: torch.Tensor,
    pf_logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


def backward_step_ou(
    s: torch.Tensor | None,  # state at time t; if None, we sample
    s_next: torch.Tensor,  # state at time t + \Delta t
    step: int,  # step at time t
    num_steps: int,
    pb_mean_correction: torch.Tensor,
    pb_var_correction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError
