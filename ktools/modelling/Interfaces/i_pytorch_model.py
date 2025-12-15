import torch
from abc import ABC, abstractmethod


class IPytorchModel(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.tensor:
        pass
