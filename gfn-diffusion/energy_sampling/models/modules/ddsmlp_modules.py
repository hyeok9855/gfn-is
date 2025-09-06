from typing import Callable, cast
import torch
from torch import nn

from models.modules.mlp_modules import MLPModule, ParamGroups


class DDSMLPModule(MLPModule):
    def initialize(self):
        self.timestep_phase = nn.Parameter(torch.zeros(self.harmonics_dim)[None])
        self.register_buffer(
            "pe", torch.linspace(start=0.1, end=100, steps=self.harmonics_dim)[None]
        )

        self.time_coder_state = nn.Sequential(
            nn.Linear(2 * self.harmonics_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.t_emb_dim),
        )

        self.time_coder_grad = None
        if self.lp:
            self.time_coder_grad = nn.Sequential(
                nn.Linear(2 * self.harmonics_dim, self.hidden_dim),
                *[
                    nn.Sequential(
                        nn.GELU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                    )
                    for _ in range(self.lgv_layers - 1)
                ],
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.lgv_out_dim),
            )
            if self.zero_init:
                self.time_coder_grad[-1].weight.data.fill_(1e-8)
                self.time_coder_grad[-1].bias.data.fill_(0.01)

        self.state_time_net = nn.Sequential(
            nn.Linear(self.ndim + self.t_emb_dim, self.hidden_dim),
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU())
                for _ in range(self.joint_layers - 1)
            ],
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        if self.zero_init:
            self.state_time_net[-1].weight.data.fill_(1e-8)
            self.state_time_net[-1].bias.data.fill_(0.0)

        self.bwd_t_model = self.bwd_s_model = self.bwd_joint_model = None
        if self.learn_pb:  # TODO: implement backward correction
            raise NotImplementedError("Backward correction is not implemented for DDSMLPModule!")

        if self.conditional_flow_model:  # TODO: implement conditional flow
            raise NotImplementedError("Conditional flow is not implemented for DDSMLPModule!")
        else:
            self.flow_model = torch.nn.Parameter(torch.tensor(0.0))

    def get_param_groups(self) -> ParamGroups:
        forward_params = []
        forward_params += [self.timestep_phase]
        forward_params += list(self.time_coder_state.parameters())
        forward_params += list(self.state_time_net.parameters())

        backward_params = []

        assert isinstance(self.flow_model, nn.Parameter)
        flow_params = [self.flow_model]

        lgv_params = []
        if self.time_coder_grad is not None:
            lgv_params += list(self.time_coder_grad.parameters())

        return ParamGroups(
            forward_params=forward_params,
            backward_params=backward_params,
            flow_params=flow_params,
            lgv_params=lgv_params,
        )

    def predict_forward(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        grad_logr_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sin_embed_cond = ((t.unsqueeze(1) * self.pe) + self.timestep_phase).sin()  # type: ignore
        cos_embed_cond = ((t.unsqueeze(1) * self.pe) + self.timestep_phase).cos()  # type: ignore
        time_array_emb = torch.cat([sin_embed_cond, cos_embed_cond], dim=-1)

        t_net1 = self.time_coder_state(time_array_emb)

        extended_input = torch.cat([s, t_net1], dim=-1)
        out_state = self.state_time_net(extended_input)

        mean, logvar = self.get_gaussian_params(
            out_state, s, t, grad_logr_fn, time_array_emb=time_array_emb
        )
        flow = self.predict_flow(s=s, t=t)  # TODO: support conditional flow
        return mean, logvar, flow

    def get_lp_scaling(
        self, t: torch.Tensor, time_array_emb: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        assert self.time_coder_grad is not None
        return self.time_coder_grad(time_array_emb)
