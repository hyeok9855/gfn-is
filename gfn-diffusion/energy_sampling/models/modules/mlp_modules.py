import math
from typing import Callable

import torch
from torch import nn

from models.modules.base import BaseModule


class MLPModule(BaseModule):
    def __init__(
        self,
        ndim: int,
        harmonics_dim: int,
        t_emb_dim: int,
        s_emb_dim: int,
        hidden_dim: int,
        out_dim: int,
        flow_harmonics_dim: int = 64,
        flow_t_emb_dim: int = 64,
        flow_s_emb_dim: int = 64,
        flow_hidden_dim: int = 64,
        lp: bool = False,
        clipping: bool = False,
        lgv_clip: float = 1e2,
        gfn_clip: float = 1e4,
        pb_scale_range: float = 1.0,
        lv_out_dim: int = 1,
        lp_scaling_per_dimension: bool = True,
        conditional_flow_model: bool = False,
        share_embeddings: bool = False,
        learn_pb: bool = False,
        lgv_layers: int = 3,
        joint_layers: int = 2,
        flow_layers: int = 2,
        zero_init: bool = False,
        device=torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.ndim = ndim
        self.lp = lp

        self.clipping = clipping
        self.lgv_clip = lgv_clip
        self.gfn_clip = gfn_clip

        self.lp_scaling_per_dimension = lp_scaling_per_dimension

        self.conditional_flow_model = conditional_flow_model
        self.share_embeddings = share_embeddings

        self.learn_pb = learn_pb
        self.pb_scale_range = pb_scale_range

        self.bwd_t_model = self.bwd_s_model = self.bwd_joint_model = self.lp_scaling_model = None

        self.t_model = TimeEncoding(harmonics_dim, t_emb_dim, hidden_dim)
        self.s_model = StateEncoding(ndim, hidden_dim, s_emb_dim)
        self.joint_model = JointPolicy(
            s_emb_dim, t_emb_dim, hidden_dim, out_dim, joint_layers, zero_init
        )
        if learn_pb:
            self.bwd_t_model = TimeEncoding(harmonics_dim, t_emb_dim, hidden_dim)
            self.bwd_s_model = StateEncoding(ndim, hidden_dim, s_emb_dim)
            self.bwd_joint_model = JointPolicy(
                s_emb_dim, t_emb_dim, hidden_dim, out_dim, joint_layers, zero_init
            )

        if self.conditional_flow_model:
            self.t_model_flow = self.s_model_flow = None
            if not self.share_embeddings:
                self.t_model_flow = TimeEncoding(
                    flow_harmonics_dim, flow_t_emb_dim, flow_hidden_dim
                )
                self.s_model_flow = StateEncoding(ndim, flow_hidden_dim, flow_s_emb_dim)
            else:
                flow_t_emb_dim, flow_s_emb_dim = t_emb_dim, s_emb_dim
            self.flow_model = FlowModel(flow_s_emb_dim, flow_t_emb_dim, hidden_dim, 1, flow_layers)
        else:
            self.flow_model = torch.nn.Parameter(torch.tensor(0.0).to(device))

        if self.lp:
            self.lp_scaling_model = LangevinScalingModel(
                s_emb_dim, t_emb_dim, hidden_dim, lv_out_dim, lgv_layers, zero_init
            )

    def predict_forward(
        self, s: torch.Tensor, t: torch.Tensor, logr_fn: Callable | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            scale = self.lp_scaling_model(s_emb, t_emb)
            s_new[..., : self.ndim] += scale * grad_log_r

        if self.clipping:
            s_new = torch.clip(s_new, -self.gfn_clip, self.gfn_clip)
        return s_new, flow.squeeze(-1)

    def predict_backward(self, s_next: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor | None:
        out = None
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
        return out


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
