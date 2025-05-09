from typing import Callable
import torch
from torch import nn

from models.modules.mlp_modules import MLPModule


class PISMLPModule(MLPModule):
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
        super().__init__(
            ndim,
            harmonics_dim,
            t_emb_dim,
            s_emb_dim,
            hidden_dim,
            out_dim,
            flow_harmonics_dim,
            flow_t_emb_dim,
            flow_s_emb_dim,
            flow_hidden_dim,
            lp,
            clipping,
            lgv_clip,
            gfn_clip,
            pb_scale_range,
            lv_out_dim,
            lp_scaling_per_dimension,
            conditional_flow_model,
            share_embeddings,
            learn_pb,
            lgv_layers,
            joint_layers,
            flow_layers,
            zero_init,
            device,
        )

        assert (
            s_emb_dim == t_emb_dim
        ), "Dimensionality of state embedding and time embedding should be the same!"

        self.t_model = TimeEncodingPIS(harmonics_dim, t_emb_dim, hidden_dim)
        self.s_model = StateEncodingPIS(ndim, s_emb_dim)
        self.joint_model = JointPolicyPIS(s_emb_dim, hidden_dim, out_dim, joint_layers, zero_init)

        if learn_pb:
            self.bwd_t_model = TimeEncodingPIS(harmonics_dim, t_emb_dim, hidden_dim)
            self.bwd_s_model = StateEncodingPIS(ndim, s_emb_dim)
            self.bwd_joint_model = JointPolicyPIS(
                s_emb_dim, hidden_dim, out_dim, joint_layers, zero_init
            )

        if self.conditional_flow_model:
            self.t_model_flow = self.s_model_flow = None
            if not self.share_embeddings:
                assert flow_t_emb_dim == flow_s_emb_dim
                self.t_model_flow = TimeEncodingPIS(
                    flow_harmonics_dim, flow_t_emb_dim, flow_hidden_dim
                )
                self.s_model_flow = StateEncodingPIS(ndim, flow_s_emb_dim)
            else:
                flow_t_emb_dim, flow_s_emb_dim = t_emb_dim, s_emb_dim
            self.flow_model = FlowModelPIS(flow_s_emb_dim, flow_hidden_dim, 1, flow_layers)
        else:
            self.flow_model = torch.nn.Parameter(torch.tensor(0.0).to(device))

        if self.lp:
            self.lp_scaling_model = LangevinScalingModelPIS(
                t_emb_dim, hidden_dim, lv_out_dim, lgv_layers, zero_init
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
            scale = self.lp_scaling_model(t)  # !!! Note here !!!
            s_new[..., : self.ndim] += scale * grad_log_r

        if self.clipping:
            s_new = torch.clip(s_new, -self.gfn_clip, self.gfn_clip)
        return s_new, flow.squeeze(-1)


class TimeEncodingPIS(nn.Module):
    def __init__(self, harmonics_dim: int, t_emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.timestep_phase = nn.Parameter(torch.randn(harmonics_dim)[None])
        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, t_emb_dim),
        )
        self.register_buffer("pe", torch.linspace(start=0.1, end=100, steps=harmonics_dim)[None])

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            t: torch.Tensor
        """
        t_sin = ((t.unsqueeze(1) * self.pe) + self.timestep_phase).sin()  # type: ignore
        t_cos = ((t.unsqueeze(1) * self.pe) + self.timestep_phase).cos()  # type: ignore
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.t_model(t_emb)


class StateEncodingPIS(nn.Module):
    def __init__(self, ndim: int, s_emb_dim: int) -> None:
        super().__init__()

        self.s_model = nn.Linear(ndim, s_emb_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.s_model(s)


class JointPolicyPIS(nn.Module):
    def __init__(
        self,
        s_emb_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        zero_init: bool = False,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.GELU(),
            nn.Linear(s_emb_dim, hidden_dim),
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
        return self.model(s_emb + t_emb)


class FlowModelPIS(nn.Module):
    def __init__(
        self,
        s_emb_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.GELU(),
            nn.Linear(s_emb_dim, hidden_dim),
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers - 1)
            ],
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, s_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.model(s_emb + t_emb)


class LangevinScalingModelPIS(nn.Module):
    def __init__(
        self,
        t_emb_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        zero_init: bool = False,
    ) -> None:
        super().__init__()

        pe = torch.linspace(start=0.1, end=100, steps=t_emb_dim)[None]

        self.timestep_phase = nn.Parameter(torch.randn(t_emb_dim)[None])

        self.lgv_model = nn.Sequential(
            nn.Linear(2 * t_emb_dim, hidden_dim),
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

        self.register_buffer("pe", pe)

        if zero_init:
            self.lgv_model[-1].weight.data.fill_(0.0)
            self.lgv_model[-1].bias.data.fill_(0.01)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_sin = ((t.unsqueeze(1) * self.pe) + self.timestep_phase).sin()  # type: ignore
        t_cos = ((t.unsqueeze(1) * self.pe) + self.timestep_phase).cos()  # type: ignore
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.lgv_model(t_emb)
