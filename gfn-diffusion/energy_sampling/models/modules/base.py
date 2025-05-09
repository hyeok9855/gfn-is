from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass
class ParamGroups:
    forward_params: list[nn.Parameter]
    backward_params: list[nn.Parameter]
    flow_params: list[nn.Parameter]


class BaseModule(nn.Module, ABC):
    def __init__(self, conditional_flow_model: bool) -> None:
        super().__init__()
        self.conditional_flow_model = conditional_flow_model

    @abstractmethod
    def get_param_groups(self) -> ParamGroups:
        raise NotImplementedError

    @abstractmethod
    def predict_forward(
        self, s: torch.Tensor, t: torch.Tensor, logr_fn: Callable | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # mean, logvar, flow
        raise NotImplementedError

    def predict_backward(
        self, s_next: torch.Tensor, t_next: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:  # bwd_mean_correction, bwd_var_correction
        return torch.ones_like(s_next), torch.ones_like(s_next)
