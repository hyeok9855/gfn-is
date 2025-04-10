import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from models.architectures import (FlowModel, FlowModelPIS, JointPolicy, JointPolicyPIS,
                                  LangevinScalingModel, LangevinScalingModelPIS, StateEncoding,
                                  StateEncodingPIS, TimeEncoding, TimeEncodingPIS)

logtwopi = math.log(2 * math.pi)


class GFN(nn.Module):
    def __init__(
        self,
        ndim: int,
        harmonics_dim: int,
        t_emb_dim: int,
        s_emb_dim: int,
        hidden_dim: int,
        log_var_range: float = 4.0,
        t_scale: float = 1.0,
        lp: bool = False,
        learned_variance: bool = True,
        partial_energy: bool = False,
        clipping: bool = False,
        lgv_clip: float = 1e2,
        gfn_clip: float = 1e4,
        pb_scale_range: float = 1.0,
        lp_scaling_per_dimension: bool = True,
        conditional_flow_model: bool = False,
        share_embeddings: bool = False,
        learn_pb: bool = False,
        pis_architectures: bool = False,
        lgv_layers: int = 3,
        joint_layers: int = 2,
        zero_init: bool = False,
        device=torch.device("cuda"),
    ) -> None:
        super(GFN, self).__init__()
        self.dim = ndim
        self.harmonics_dim = harmonics_dim
        self.t_emb_dim = t_emb_dim
        self.s_emb_dim = s_emb_dim

        self.lp = lp
        self.learned_variance = learned_variance
        self.partial_energy = partial_energy
        self.t_scale = t_scale

        self.clipping = clipping
        self.lgv_clip = lgv_clip
        self.gfn_clip = gfn_clip

        self.lp_scaling_per_dimension = lp_scaling_per_dimension
        self.conditional_flow_model = conditional_flow_model
        self.share_embeddings = share_embeddings
        self.learn_pb = learn_pb
        self.pb_scale_range = pb_scale_range

        self.pis_architectures = pis_architectures
        self.lgv_layers = lgv_layers
        self.joint_layers = joint_layers

        self.pf_std_per_traj = np.sqrt(self.t_scale)
        self.log_var_range = log_var_range
        self.device = device

        out_dim = 2 * ndim if self.learned_variance else ndim
        lv_out_dim = ndim if self.lp_scaling_per_dimension else 1

        self.back_model = self.lp_scaling_model = None

        if self.pis_architectures:
            assert s_emb_dim == t_emb_dim, print(
                "Dimensionality of state embedding and time embedding should be the same!"
            )  # Why?

            self.t_model = TimeEncodingPIS(harmonics_dim, t_emb_dim, hidden_dim)
            self.s_model = StateEncodingPIS(ndim, s_emb_dim)
            self.joint_model = JointPolicyPIS(
                s_emb_dim, hidden_dim, out_dim, joint_layers, zero_init
            )

            if learn_pb:
                self.back_model = JointPolicyPIS(
                    s_emb_dim, hidden_dim, out_dim, joint_layers, zero_init
                )

            if self.conditional_flow_model:
                self.t_model_flow = self.s_model_flow = None
                if not self.share_embeddings:
                    self.t_model_flow = TimeEncodingPIS(harmonics_dim, t_emb_dim, hidden_dim)
                    self.s_model_flow = StateEncodingPIS(ndim, s_emb_dim)
                self.flow_model = FlowModelPIS(s_emb_dim, hidden_dim, 1, joint_layers)
            else:
                self.flow_model = torch.nn.Parameter(torch.tensor(0.0).to(self.device))

            if self.lp:
                self.lp_scaling_model = LangevinScalingModelPIS(
                    t_emb_dim, hidden_dim, lv_out_dim, lgv_layers, zero_init
                )

        else:

            self.t_model = TimeEncoding(harmonics_dim, t_emb_dim, hidden_dim)
            self.s_model = StateEncoding(ndim, hidden_dim, s_emb_dim)
            self.joint_model = JointPolicy(s_emb_dim, t_emb_dim, hidden_dim, out_dim, zero_init)
            if learn_pb:
                self.back_model = JointPolicy(s_emb_dim, t_emb_dim, hidden_dim, out_dim, zero_init)

            if self.conditional_flow_model:
                self.flow_model = FlowModel(s_emb_dim, t_emb_dim, hidden_dim, 1)
            else:
                self.flow_model = torch.nn.Parameter(torch.tensor(0.0).to(self.device))

            if self.lp:
                self.lp_scaling_model = LangevinScalingModel(
                    s_emb_dim, t_emb_dim, hidden_dim, lv_out_dim, zero_init
                )

    def split_params(self, tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.learned_variance:
            mean = tensor
            logvar = torch.zeros_like(mean)
        else:
            mean, logvar = torch.chunk(tensor, 2, dim=-1)
            logvar = torch.tanh(logvar) * self.log_var_range
        return mean, logvar + np.log(self.pf_std_per_traj) * 2.0

    def predict_next_state(self, s, t, log_r_fn: Callable | None = None):
        if self.lp:
            assert log_r_fn is not None
            s.requires_grad_(True)
            with torch.enable_grad():
                grad_log_r = torch.autograd.grad(log_r_fn(s).sum(), s)[0].detach()
                grad_log_r = torch.nan_to_num(grad_log_r)
                if self.clipping:
                    grad_log_r = torch.clip(grad_log_r, -self.lgv_clip, self.lgv_clip)

        s_emb = self.s_model(s)
        t_emb = self.t_model(t)
        s_new = self.joint_model(s_emb, t_emb)

        if self.conditional_flow_model:
            if not self.share_embeddings:
                assert self.s_model_flow is not None and self.t_model_flow is not None
                s_emb_flow = self.s_model_flow(s)
                t_emb_flow = self.t_model_flow(t)
            else:
                s_emb_flow = s_emb
                t_emb_flow = t_emb
            flow = self.flow_model(s_emb_flow, t_emb_flow).squeeze(-1)
        else:
            flow = self.flow_model  # A learnable scalar

        if self.lp:
            assert self.lp_scaling_model is not None
            if self.pis_architectures:
                scale = self.lp_scaling_model(t)
            else:
                scale = self.lp_scaling_model(s_emb, t_emb)
            s_new[..., : self.dim] += scale * grad_log_r

        if self.clipping:
            s_new = torch.clip(s_new, -self.gfn_clip, self.gfn_clip)
        return s_new, flow.squeeze(-1)

    def get_trajectory_fwd(
        self,
        s: torch.Tensor,
        ts: torch.Tensor,
        exploration_std=0.0,
        log_r_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        pis=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = s.shape[0]
        T = ts.shape[1] - 1

        logpf = torch.zeros((bsz, T), device=self.device)
        logpb = torch.zeros((bsz, T), device=self.device)
        logf = torch.zeros((bsz, T + 1), device=self.device)
        logpf_exp = torch.zeros((bsz, T), device=self.device)
        states = torch.zeros((bsz, T + 1, self.dim), device=self.device)

        for i in range(T):
            dts = (ts[:, i + 1] - ts[:, i]).unsqueeze(1)

            pfs, flow = self.predict_next_state(s, ts[:, i], log_r_fn)
            pfmean, pflogvars = self.split_params(pfs)

            logf[:, i] = flow
            if self.partial_energy:
                assert log_r_fn is not None
                ref_log_var = (self.t_scale * ts[:, max(1, i)]).log()
                import pdb

                pdb.set_trace()  # TODO: Check if this is correct
                log_p_ref = -0.5 * (logtwopi + ref_log_var + (-ref_log_var).exp() * (s**2)).sum(1)
                logf[:, i] += (1 - ts[:, i]) * log_p_ref + ts[:, i] * log_r_fn(s)

            # PIS requires gradients w.r.t. the parameters
            if pis:
                assert exploration_std == 0.0
                pfmean_sample = pfmean
                pflogvars_sample = pflogvars
            else:
                pfmean_sample = pfmean.detach()
                pflogvars_sample = pflogvars.detach()
                # Add exploration noise
                if exploration_std > 0.0:
                    add_log_var = (
                        torch.ones_like(pflogvars_sample) * (exploration_std / dts.sqrt()).log() * 2
                    )
                    pflogvars_sample = torch.logaddexp(pflogvars_sample, add_log_var)

            s_ = (
                s
                + dts * pfmean_sample
                + dts.sqrt()
                * (pflogvars_sample / 2).exp()
                * torch.randn_like(s, device=self.device)
            )

            noise = ((s_ - s) - dts * pfmean) / (dts.sqrt() * (pflogvars / 2).exp())
            logpf[:, i] = -0.5 * (noise**2 + logtwopi + dts.log() + pflogvars).sum(1)

            if exploration_std > 0.0:
                noise_exp = ((s_ - s) - dts * pfmean_sample) / (
                    dts.sqrt() * (pflogvars_sample / 2).exp()
                )
                logpf_exp[:, i] = -0.5 * (
                    noise_exp**2 + logtwopi + dts.log() + pflogvars_sample
                ).sum(1)
            else:
                logpf_exp[:, i] = logpf[:, i].detach()

            if self.learn_pb:
                assert self.back_model is not None
                t = self.t_model(ts[:, i + 1])
                pbs = self.back_model(self.s_model(s_), t)
                dmean, dvar = torch.chunk(pbs, 2, dim=-1)
                back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
                back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
            else:
                back_mean_correction, back_var_correction = torch.ones_like(s_), torch.ones_like(s_)

            if i > 0:
                back_mean = s_ - s_ * dts / (ts[:, i + 1]).unsqueeze(1) * back_mean_correction
                back_var = (
                    (self.pf_std_per_traj**2)
                    * dts
                    * (ts[:, i] / ts[:, i + 1]).unsqueeze(1)
                    * back_var_correction
                )
                noise_backward = (s - back_mean) / back_var.sqrt()
                logpb[:, i] = -0.5 * (noise_backward**2 + logtwopi + back_var.log()).sum(1)

            s = s_
            states[:, i + 1] = s

        logpf_exp = logpf_exp if exploration_std > 0.0 else logpf

        return states, logpf, logpb, logf, logpf_exp

    def get_trajectory_bwd(
        self,
        s: torch.Tensor,
        ts: torch.Tensor,
        log_r_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = s.shape[0]
        T = ts.shape[1] - 1

        logpf = torch.zeros((bsz, T), device=self.device)
        logpb = torch.zeros((bsz, T), device=self.device)
        logf = torch.zeros((bsz, T + 1), device=self.device)
        states = torch.zeros((bsz, T + 1, self.dim), device=self.device)
        states[:, -1] = s

        for i in range(T):
            dts = (ts[:, T - i] - ts[:, T - i - 1]).unsqueeze(1)

            if i < T - 1:
                if self.learn_pb:
                    assert self.back_model is not None
                    t = self.t_model(ts[:, T - i])
                    pbs = self.back_model(self.s_model(s), t)
                    dmean, dvar = torch.chunk(pbs, 2, dim=-1)
                    back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
                    back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
                else:
                    back_mean_correction, back_var_correction = torch.ones_like(s), torch.ones_like(
                        s
                    )

                mean = s - s * dts / ts[:, T - i].unsqueeze(1) * back_mean_correction
                var = (
                    (self.pf_std_per_traj**2)
                    * dts
                    * (ts[:, T - i - 1] / ts[:, T - i]).unsqueeze(1)
                    * back_var_correction
                )

                s_ = mean.detach() + var.sqrt().detach() * torch.randn_like(s, device=self.device)
                noise_backward = (s_ - mean) / var.sqrt()
                logpb[:, T - i - 1] = -0.5 * (noise_backward**2 + logtwopi + var.log()).sum(1)
            else:
                s_ = torch.zeros_like(s)

            pfs, flow = self.predict_next_state(s_, ts[:, T - i - 1], log_r_fn)
            pfmean, pflogvars = self.split_params(pfs)

            logf[:, T - i - 1] = flow
            if self.partial_energy:
                assert log_r_fn is not None
                ref_log_var = (self.t_scale * ts[:, max(1, T - i - 1)]).log()
                log_p_ref = -0.5 * (logtwopi + ref_log_var + (-ref_log_var).exp() * (s**2)).sum(1)
                logf[:, T - i - 1] += ts[:, T - i - 1] * log_p_ref + ts[:, i + 1] * log_r_fn(s)

            noise = ((s - s_) - dts * pfmean) / (dts.sqrt() * (pflogvars / 2).exp())
            logpf[:, T - i - 1] = -0.5 * (noise**2 + logtwopi + dts.log() + pflogvars).sum(1)

            s = s_
            states[:, T - i - 1] = s

        return states, logpf, logpb, logf

    # Not used for now
    def forward(self, s, exploration_std=0.0, log_r_fn: Callable | None = None):
        raise NotImplementedError
