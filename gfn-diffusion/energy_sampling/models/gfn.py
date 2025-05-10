import math

import torch
import torch.nn as nn

from energies import BaseEnergy, LennardJones
from models.modules import BaseModule
from utils.particle_system import remove_mean

logtwopi = math.log(2 * math.pi)


class GFN(nn.Module):
    def __init__(
        self,
        energy: BaseEnergy,
        module: BaseModule,
        t_scale: float = 1.0,
        partial_energy: bool = False,
        learn_beta_T: int = 0,
        state_reduce_mean: bool = False,
        device=torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.energy = energy
        self.pred_module = module
        self.t_scale = t_scale
        self.partial_energy = partial_energy
        self.state_reduce_mean = state_reduce_mean
        self.device = device

        self.beta_model = None
        if learn_beta_T > 0:
            self.beta_model = torch.nn.Parameter(
                torch.cat(
                    [
                        torch.tensor([-float("inf")], device=self.device),
                        torch.ones(learn_beta_T, device=self.device) * -1.0,
                    ],
                )
            )
            self.softplus = nn.Softplus()

    def forward_step(
        self,
        s: torch.Tensor,  # state at time t
        s_next: torch.Tensor | None,  # state at time t + \Delta t; if None, we sample
        t: torch.Tensor,  # time t
        t_next: torch.Tensor,  # time t + \Delta t
        pf_mean: torch.Tensor,
        pf_logvar: torch.Tensor,
        pis: bool = False,
        epsilon: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # s.shape = (bsz, ndim)
        # s_next.shape = (bsz, ndim)
        # t.shape = (bsz,)
        # t_next.shape = (bsz,)
        # pf_mean.shape = (bsz, ndim)
        # pf_logvar.shape = (bsz, ndim)

        dts = (t_next - t).unsqueeze(1)

        pf_logvar = pf_logvar + math.log(self.t_scale)

        # PIS requires gradients w.r.t. the parameters
        if pis:
            assert epsilon == 0.0
            pf_mean_sample = pf_mean
            pf_logvar_sample = pf_logvar
        else:
            pf_mean_sample = pf_mean.detach()
            pf_logvar_sample = pf_logvar.detach()
            # Add exploration noise
            if epsilon > 0.0:
                pf_logvar_sample = pf_logvar_sample + math.log(1.0 + epsilon)

        if s_next is None:
            s_next = (
                s
                + dts * pf_mean_sample
                + dts.sqrt()
                * (pf_logvar_sample / 2).exp()
                * torch.randn_like(s, device=self.device)
            )

            if self.state_reduce_mean:
                assert isinstance(self.energy, LennardJones)
                s_next = remove_mean(s_next, self.energy.n_particles, self.energy.spatial_dim)

        noise = ((s_next - s) - dts * pf_mean) / (dts.sqrt() * (pf_logvar / 2).exp())
        log_pfs = -0.5 * (noise**2 + logtwopi + dts.log() + pf_logvar).sum(1)

        if epsilon > 0.0:
            noise_exp = ((s_next - s) - dts * pf_mean_sample) / (
                dts.sqrt() * (pf_logvar_sample / 2).exp()
            )
            log_pfs_exp = -0.5 * (noise_exp**2 + logtwopi + dts.log() + pf_logvar_sample).sum(1)
        else:
            log_pfs_exp = log_pfs.detach()

        return s_next, log_pfs, log_pfs_exp

    def backward_step(
        self,
        s: torch.Tensor | None,  # state at time t; if None, we sample
        s_next: torch.Tensor,  # state at time t + \Delta t
        t: torch.Tensor,  # t
        t_next: torch.Tensor,  # t + \Delta t
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # s.shape = (bsz, ndim)
        # s_next.shape = (bsz, ndim)
        # t.shape = (bsz,)
        # t_next.shape = (bsz,)

        dts = (t_next - t).unsqueeze(1)

        mean_correction, var_correction = self.pred_module.predict_backward(s_next, t_next)
        back_mean = s_next - s_next * dts / (t_next).unsqueeze(1) * mean_correction
        back_var = self.t_scale * dts * (t / t_next).unsqueeze(1) * var_correction

        if s is None:
            s = torch.zeros_like(s_next)
            s[t != 0] = back_mean.detach()[t != 0] + back_var.sqrt().detach()[
                t != 0
            ] * torch.randn_like(s_next[t != 0])

            if self.state_reduce_mean:
                assert isinstance(self.energy, LennardJones)
                s = remove_mean(s, self.energy.n_particles, self.energy.spatial_dim)

        noise_backward = (s - back_mean) / back_var.sqrt()
        log_pbs = torch.zeros_like(back_mean[:, 0])
        log_pbs[t != 0] = -0.5 * (
            noise_backward[t != 0] ** 2 + logtwopi + back_var[t != 0].log()
        ).sum(1)
        return s, log_pbs

    def get_partial_energy(
        self,
        states: torch.Tensor,  # (bsz, T', ndim)
        ts: torch.Tensor,  # (bsz, T')
    ) -> torch.Tensor:
        assert self.partial_energy
        bsz = states.shape[0]

        if self.beta_model is not None:
            betas = self.softplus(self.beta_model).cumsum(0)
            betas = betas / betas[-1]
            betas = betas.quantile(ts.flatten()).reshape(bsz, -1)
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
        s: torch.Tensor,
        ts: torch.Tensor,
        epsilon: float = 0.0,
        pis=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = s.shape[0]
        T = ts.shape[1] - 1

        log_pfs = torch.zeros((bsz, T), device=self.device)
        log_pbs = torch.zeros((bsz, T), device=self.device)
        log_fs = torch.zeros((bsz, T + 1), device=self.device)
        log_pfs_exp = torch.zeros((bsz, T), device=self.device)
        states = torch.zeros((bsz, T + 1, self.energy.ndim), device=self.device)
        states[:, 0] = s

        for i in range(T):
            pf_mean, pf_logvar, flow = self.pred_module.predict_forward(
                s, ts[:, i], self.energy.log_reward
            )

            if self.pred_module.conditional_flow_model or i == 0:
                log_fs[:, i] = flow

            s_, log_pfs[:, i], log_pfs_exp[:, i] = self.forward_step(
                s, None, ts[:, i], ts[:, i + 1], pf_mean, pf_logvar, pis, epsilon
            )
            if i > 0:
                _, log_pbs[:, i] = self.backward_step(s, s_, ts[:, i], ts[:, i + 1])

            s = s_
            states[:, i + 1] = s

        # Assign the terminal reward
        # Set terminal reward based on whether we need gradients for PIS loss
        with torch.enable_grad() if pis else torch.no_grad():
            log_fs[:, -1] = self.energy.log_reward(states[:, -1])

        if self.partial_energy:
            log_fs[:, 1:-1] += self.get_partial_energy(states[:, 1:-1], ts[:, 1:-1])

        log_pfs_exp = log_pfs_exp if epsilon > 0.0 else log_pfs

        return states, log_pfs, log_pbs, log_fs, log_pfs_exp

    def get_trajectory_bwd(
        self,
        s: torch.Tensor,
        ts: torch.Tensor,  # (bsz, T)
        log_r: torch.Tensor,  # (bsz,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = s.shape[0]
        T = ts.shape[1] - 1

        log_pfs = torch.zeros((bsz, T), device=self.device)
        log_pbs = torch.zeros((bsz, T), device=self.device)
        log_fs = torch.zeros((bsz, T + 1), device=self.device)
        states = torch.zeros((bsz, T + 1, self.energy.ndim), device=self.device)
        states[:, -1] = s

        for i in range(T):
            if i < T - 1:
                s_, log_pbs[:, T - i - 1] = self.backward_step(
                    None, s, ts[:, T - i - 1], ts[:, T - i]
                )
            else:
                s_ = torch.zeros_like(s)

            pf_mean, pf_logvar, flow = self.pred_module.predict_forward(
                s_, ts[:, T - i - 1], self.energy.log_reward
            )

            if self.pred_module.conditional_flow_model or i == T - 1:
                log_fs[:, T - i - 1] = flow

            _, log_pfs[:, T - i - 1], _ = self.forward_step(
                s_, s, ts[:, T - i - 1], ts[:, T - i], pf_mean, pf_logvar
            )

            s = s_
            states[:, T - i - 1] = s

        log_fs[:, -1] = log_r

        if self.partial_energy:
            log_fs[:, 1:-1] += self.get_partial_energy(states[:, 1:-1], ts[:, 1:-1])

        return states, log_pfs, log_pbs, log_fs

    def get_trajectory_fwd_and_bwd(
        self,
        s: torch.Tensor,  # (bsz, ndim)
        ts: torch.Tensor,  # (bsz, T+1)
        curr_t: torch.Tensor,  # (bsz)
        epsilon: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Construct complete trajectories both ways
        # 1. Backward sampling from intermediate_ts to 0
        # 2. Forward sampling from intermediate_ts to T
        # 3. Concatenate the two trajectories
        bsz = s.shape[0]
        T = ts.shape[1] - 1
        arange = torch.arange(bsz)

        t_idx_intermediate = torch.where(ts == curr_t.unsqueeze(1))[1]
        t_idx = t_idx_intermediate.clone()
        t_idx_next = t_idx.clone()
        is_backward = t_idx > 0

        log_pfs = torch.zeros((bsz, T), device=self.device)
        log_pbs = torch.zeros((bsz, T), device=self.device)
        log_fs = torch.zeros((bsz, T + 1), device=self.device)
        log_pfs_exp = torch.zeros((bsz, T), device=self.device)
        states = torch.zeros((bsz, T + 1, self.energy.ndim), device=self.device)
        states[arange, t_idx_intermediate] = s

        if (is_intermediate := t_idx_intermediate != T).any():
            # For non-terminal states, we should predict the flow at the current timestep
            _, _, flow = self.pred_module.predict_forward(
                s[is_intermediate], curr_t[is_intermediate], self.energy.log_reward
            )
            log_fs[is_intermediate, t_idx_intermediate[is_intermediate]] = flow
            # The terminal states rewards will be assigned later

        t1 = torch.zeros_like(curr_t)
        t2 = torch.zeros_like(curr_t)
        for _ in range(T):
            t_idx_next[is_backward] = t_idx[is_backward] - 1
            t_idx_next[~is_backward] = t_idx[~is_backward] + 1
            t1_idx = torch.min(t_idx, t_idx_next)
            t2_idx = torch.max(t_idx, t_idx_next)
            t1 = ts[arange, t1_idx]
            t2 = ts[arange, t2_idx]

            t1_idx_bwd = t1_idx[is_backward]
            t1_idx_fwd = t1_idx[~is_backward]
            t2_idx_bwd = t2_idx[is_backward]
            t2_idx_fwd = t2_idx[~is_backward]

            t1_bwd = t1[is_backward]
            t1_fwd = t1[~is_backward]
            t2_bwd = t2[is_backward]
            t2_fwd = t2[~is_backward]

            # Fill the states[is_backward, t1_idx[is_backward]]
            if is_backward.any():
                states_bwd_t1, log_pb_bwd = self.backward_step(
                    None, states[is_backward, t2_idx_bwd], t1_bwd, t2_bwd
                )
                states[is_backward, t1_idx_bwd] = states_bwd_t1
                log_pbs[is_backward, t1_idx_bwd] = log_pb_bwd

            # Forward pass with the one-step forward state of t1
            pf_mean, pf_logvars, flow = self.pred_module.predict_forward(
                states[arange, t1_idx], t1, self.energy.log_reward
            )
            log_fs[arange, t1_idx] = flow

            if (~is_backward).any():
                # Fill the states[~is_backward, t2_idx[~is_backward]]
                states_fwd_t2, log_pf_fwd, log_pf_fwd_exp = self.forward_step(
                    states[~is_backward, t1_idx_fwd],
                    None,
                    t1_fwd,
                    t2_fwd,
                    pf_mean[~is_backward],
                    pf_logvars[~is_backward],
                    pis=False,
                    epsilon=epsilon,
                )
                states[~is_backward, t2_idx_fwd] = states_fwd_t2
                log_pfs[~is_backward, t1_idx_fwd] = log_pf_fwd
                log_pfs_exp[~is_backward, t1_idx_fwd] = log_pf_fwd_exp

                _, log_pb_fwd = self.backward_step(
                    states[~is_backward, t1_idx_fwd],
                    states[~is_backward, t2_idx_fwd],
                    t1_fwd,
                    t2_fwd,
                )
                log_pbs[~is_backward, t1_idx_fwd] = log_pb_fwd

            if is_backward.any():
                _, log_pf_bwd, log_pf_bwd_exp = self.forward_step(
                    states[is_backward, t1_idx_bwd],
                    states[is_backward, t2_idx_bwd],
                    t1_bwd,
                    t2_bwd,
                    pf_mean[is_backward],
                    pf_logvars[is_backward],
                    pis=False,
                    epsilon=epsilon,
                )
                log_pfs[is_backward, t1_idx_bwd] = log_pf_bwd
                log_pfs_exp[is_backward, t1_idx_bwd] = log_pf_bwd_exp

            # Prepare for the next iteration
            is_initial = t1_idx == 0
            t_idx = t_idx_next.clone()
            t_idx[is_backward & is_initial] = t_idx_intermediate[is_backward & is_initial]
            is_backward = is_backward & (~is_initial)

        # Assign the terminal reward
        with torch.no_grad():
            log_fs[:, -1] = self.energy.log_reward(states[:, -1])

        if self.partial_energy:
            log_fs[:, 1:-1] += self.get_partial_energy(states[:, 1:-1], ts[:, 1:-1])

        return states, log_pfs, log_pbs, log_fs, log_pfs_exp
