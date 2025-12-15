import torch
import torch.nn as nn
from typing import *
from ktools.modelling.pytorch_utils.utils import get_activation


class NonLinearFeedForwardModule(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, activation: str = "gelu"
    ):
        super(NonLinearFeedForwardModule, self).__init__()
        self.ffm = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.ffm(x)
