from typing import Callable
from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseModule(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def predict_forward(
        self, s: torch.Tensor, t: torch.Tensor, logr_fn: Callable | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def predict_backward(self, s_next: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor | None:
        raise NotImplementedError
