import math

import torch
from torch import nn


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
        self, s_emb_dim: int, t_emb_dim: int, hidden_dim: int, out_dim: int, zero_init: bool = False
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(s_emb_dim + t_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
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


class LangevinScalingModel(nn.Module):
    def __init__(
        self, s_emb_dim: int, t_emb_dim: int, hidden_dim: int, out_dim: int, zero_init: bool = False
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(s_emb_dim + t_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.01)

    def forward(self, s_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([s_emb, t_emb], dim=-1))


class TimeEncodingPIS(nn.Module):
    def __init__(self, harmonics_dim: int, t_emb_dim: int, hidden_dim: int) -> None:
        super().__init__()

        pe = torch.linspace(start=0.1, end=100, steps=harmonics_dim)[None]

        self.timestep_phase = nn.Parameter(torch.randn(harmonics_dim)[None])

        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, t_emb_dim),
        )
        self.register_buffer("pe", pe)

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
