import math
from typing import Callable, cast

import torch
from torch import nn

from models.modules.base import BaseModule, ParamGroups


class MLPModule(BaseModule):
    def __init__(
        self,
        ndim: int,
        harmonics_dim: int,
        t_emb_dim: int,
        s_emb_dim: int,
        hidden_dim: int,
        joint_layers: int = 2,
        zero_init: bool = False,
        conditional_flow_model: bool = False,
        share_embeddings: bool = False,
        flow_harmonics_dim: int = 64,
        flow_t_emb_dim: int = 64,
        flow_s_emb_dim: int = 64,
        flow_hidden_dim: int = 64,
        flow_layers: int = 2,
        lp: bool = False,
        lp_scaling_per_dimension: bool = True,
        lgv_layers: int = 3,
        clipping: bool = False,
        lgv_clip: float = 1e2,
        gfn_clip: float = 1e4,
        learn_pb: bool = False,
        pb_scale_range: float = 1.0,
        learn_variance: bool = True,
        log_var_range: float = 4.0,
    ) -> None:
        super().__init__(conditional_flow_model)

        self.ndim = ndim
        self.harmonics_dim = harmonics_dim
        self.t_emb_dim = t_emb_dim
        self.s_emb_dim = s_emb_dim
        self.hidden_dim = hidden_dim
        self.joint_layers = joint_layers
        self.zero_init = zero_init

        self.conditional_flow_model = conditional_flow_model
        self.share_embeddings = share_embeddings
        self.flow_harmonics_dim = flow_harmonics_dim
        self.flow_t_emb_dim = flow_t_emb_dim
        self.flow_s_emb_dim = flow_s_emb_dim
        self.flow_hidden_dim = flow_hidden_dim
        self.flow_layers = flow_layers

        self.lp = lp
        self.lp_scaling_per_dimension = lp_scaling_per_dimension
        self.lgv_layers = lgv_layers

        self.clipping = clipping
        self.lgv_clip = lgv_clip
        self.gfn_clip = gfn_clip

        self.learn_pb = learn_pb
        self.pb_scale_range = pb_scale_range

        self.learn_variance = learn_variance
        self.log_var_range = log_var_range

        self.out_dim = 2 * ndim if learn_variance else ndim
        self.lgv_out_dim = ndim if lp_scaling_per_dimension else 1

        self.initialize()

    def initialize(self):
        self.t_model = TimeEncoding(self.harmonics_dim, self.t_emb_dim, self.hidden_dim)
        self.s_model = StateEncoding(self.ndim, self.hidden_dim, self.s_emb_dim)
        self.joint_model = JointPolicy(
            self.s_emb_dim,
            self.t_emb_dim,
            self.hidden_dim,
            self.out_dim,
            self.joint_layers,
            self.zero_init,
        )
        self.bwd_t_model = self.bwd_s_model = self.bwd_joint_model = None
        if self.learn_pb:
            self.bwd_t_model = TimeEncoding(self.harmonics_dim, self.t_emb_dim, self.hidden_dim)
            self.bwd_s_model = StateEncoding(self.ndim, self.hidden_dim, self.s_emb_dim)
            self.bwd_joint_model = JointPolicy(
                self.s_emb_dim,
                self.t_emb_dim,
                self.hidden_dim,
                self.out_dim,
                self.joint_layers,
                self.zero_init,
            )

        if self.conditional_flow_model:
            self.t_model_flow = self.s_model_flow = None
            if not self.share_embeddings:
                self.t_model_flow = TimeEncoding(
                    self.flow_harmonics_dim, self.flow_t_emb_dim, self.flow_hidden_dim
                )
                self.s_model_flow = StateEncoding(
                    self.ndim, self.flow_hidden_dim, self.flow_s_emb_dim
                )
            else:
                self.flow_t_emb_dim, self.flow_s_emb_dim = self.t_emb_dim, self.s_emb_dim
            self.flow_model = FlowModel(
                self.flow_s_emb_dim, self.flow_t_emb_dim, self.hidden_dim, 1, self.flow_layers
            )
        else:
            self.flow_model = torch.nn.Parameter(torch.tensor(0.0))

        self.lp_scaling_model = None
        if self.lp:
            self.lp_scaling_model = LangevinScalingModel(
                self.s_emb_dim,
                self.t_emb_dim,
                self.hidden_dim,
                self.lgv_out_dim,
                self.lgv_layers,
                self.zero_init,
            )

    def get_param_groups(self) -> ParamGroups:
        # Group parameters by their function
        forward_params = []
        forward_params += list(self.t_model.parameters())
        forward_params += list(self.s_model.parameters())
        forward_params += list(self.joint_model.parameters())
        if self.lp_scaling_model is not None:
            forward_params += list(self.lp_scaling_model.parameters())

        backward_params = []
        if self.bwd_joint_model is not None:
            assert self.bwd_t_model is not None and self.bwd_s_model is not None
            backward_params += list(self.bwd_t_model.parameters())
            backward_params += list(self.bwd_s_model.parameters())
            backward_params += list(self.bwd_joint_model.parameters())

        flow_params = []
        if isinstance(self.flow_model, torch.nn.Module):
            flow_params += list(self.flow_model.parameters())
            if self.t_model_flow is not None:
                flow_params += list(self.t_model_flow.parameters())
            if self.s_model_flow is not None:
                flow_params += list(self.s_model_flow.parameters())
        else:
            flow_params += [self.flow_model]

        return ParamGroups(
            forward_params=forward_params,
            backward_params=backward_params,
            flow_params=flow_params,
        )

    def predict_forward(
        self, s: torch.Tensor, t: torch.Tensor, logr_fn: Callable | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.lp:
            assert logr_fn is not None
            s.requires_grad_(True)
            with torch.enable_grad():
                grad_log_r = torch.autograd.grad(logr_fn(s).sum(), s)[0].detach()
                grad_log_r = torch.nan_to_num(grad_log_r)
                if self.clipping:
                    grad_log_r = torch.clip(grad_log_r, -self.lgv_clip, self.lgv_clip)

        s_emb = self.s_model(s)
        t_emb = self.t_model(t)
        out = self.joint_model(s_emb, t_emb)

        if self.conditional_flow_model:
            if not self.share_embeddings:
                assert self.s_model_flow is not None and self.t_model_flow is not None
                s_emb_flow = self.s_model_flow(s)
                t_emb_flow = self.t_model_flow(t)
            else:
                s_emb_flow = s_emb
                t_emb_flow = t_emb
            assert isinstance(self.flow_model, nn.Module)
            flow = self.flow_model(s_emb_flow, t_emb_flow).squeeze(-1)
        else:
            flow = self.flow_model  # A learnable scalar

        if self.lp:
            scale = self.get_lp_scaling(s_emb, t_emb, t)
            out[..., : self.ndim] += scale * grad_log_r

        if self.clipping:
            out = torch.clip(out, -self.gfn_clip, self.gfn_clip)

        if not self.learn_variance:
            mean = out
            logvar = torch.zeros_like(mean)
        else:
            mean, logvar = torch.chunk(out, 2, dim=-1)
            logvar = torch.tanh(logvar) * self.log_var_range

        return mean, logvar, cast(torch.Tensor, flow)

    def predict_backward(
        self, s_next: torch.Tensor, t_next: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.learn_pb:
            assert (
                self.bwd_t_model is not None
                and self.bwd_s_model is not None
                and self.bwd_joint_model is not None
            )
            t_emb = self.bwd_t_model(t_next)
            s_emb = self.bwd_s_model(s_next)
            out = self.bwd_joint_model(torch.cat([s_emb, t_emb], dim=-1))
            if self.clipping:
                out = torch.clip(out, -self.gfn_clip, self.gfn_clip)

            bwd_mean, bwd_var = torch.chunk(out, 2, dim=-1)
            mean_correction = 1 + bwd_mean.tanh() * self.pb_scale_range
            var_correction = 1 + bwd_var.tanh() * self.pb_scale_range
        else:
            mean_correction, var_correction = torch.ones_like(s_next), torch.ones_like(s_next)
        return mean_correction, var_correction

    def get_lp_scaling(
        self, s_emb: torch.Tensor, t_emb: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        assert self.lp_scaling_model is not None
        return self.lp_scaling_model(s_emb, t_emb)


class TimeEncoding(nn.Module):
    def __init__(self, harmonics_dim: int, t_emb_dim: int, hidden_dim: int) -> None:
        super().__init__()

        pe = torch.arange(1, harmonics_dim + 1).float().unsqueeze(0) * 2 * math.pi
        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, t_emb_dim),
            nn.GELU(),
        )
        self.register_buffer("pe", pe)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            t: torch.Tensor
        """
        t_sin = (t.unsqueeze(1) * self.pe).sin()  # type: ignore
        t_cos = (t.unsqueeze(1) * self.pe).cos()  # type: ignore
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.t_model(t_emb)


class StateEncoding(nn.Module):
    def __init__(self, ndim: int, hidden_dim: int, s_emb_dim: int) -> None:
        super().__init__()

        self.s_model = nn.Sequential(
            nn.Linear(ndim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, s_emb_dim),
            nn.GELU(),
        )

    def forward(self, s_emb: torch.Tensor) -> torch.Tensor:
        return self.s_model(s_emb)


class JointPolicy(nn.Module):
    def __init__(
        self,
        s_emb_dim: int,
        t_emb_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.GELU(),
            nn.Linear(s_emb_dim + t_emb_dim, hidden_dim),
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers - 1)
            ],
            nn.Linear(hidden_dim, out_dim),
        )
        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.0)

    def forward(self, s_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([s_emb, t_emb], dim=-1))


class FlowModel(nn.Module):
    def __init__(
        self,
        s_emb_dim: int,
        t_emb_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.GELU(),
            nn.Linear(s_emb_dim + t_emb_dim, hidden_dim),
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers - 1)
            ],
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, s_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([s_emb, t_emb], dim=-1))


class LangevinScalingModel(nn.Module):
    def __init__(
        self,
        s_emb_dim: int,
        t_emb_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        zero_init: bool = False,
    ) -> None:
        super().__init__()

        self.lgv_model = nn.Sequential(
            nn.Linear(s_emb_dim + t_emb_dim, hidden_dim),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        if zero_init:
            self.lgv_model[-1].weight.data.fill_(0.0)
            self.lgv_model[-1].bias.data.fill_(0.01)

    def forward(self, s_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.lgv_model(torch.cat([s_emb, t_emb], dim=-1))
