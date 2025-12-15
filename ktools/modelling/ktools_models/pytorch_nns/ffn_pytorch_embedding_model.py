from collections import OrderedDict
from typing import List
import torch.nn as nn
from ktools.modelling.ktools_models.pytorch_nns.base_pytorch_embedding_model import (
    BasePytorchEmbeddingModel,
)


class FFNPytorchEmbeddingModel(BasePytorchEmbeddingModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        categorical_idcs: List[int],
        categorical_sizes: List[int],
        categorical_embedding: List[int],
        activation: str,
        last_activation: str,
        num_hidden_layers: int = 1,
        largest_hidden_dim: int = 256,
        dim_decay: float = 1.0,
    ):
        self._activation = activation
        self._last_activation = last_activation
        self._largest_hidden_dim = largest_hidden_dim
        self._num_hidden_layers = num_hidden_layers
        self._dim_decay = dim_decay

        super().__init__(
            input_dim,
            output_dim,
            categorical_idcs,
            categorical_sizes,
            categorical_embedding,
        )

    def _create_model(self):
        layers = OrderedDict()
        prev_dim = self._expanded_dim
        curr_dim = self._largest_hidden_dim

        for l in range(self._num_hidden_layers):
            layers[f"layer_{l}"] = nn.Linear(prev_dim, curr_dim)
            layers[f"activation_{l}"] = self._get_activation(self._activation)
            prev_dim = curr_dim
            curr_dim = max(int(curr_dim * self._dim_decay), self._output_dim)

        layers["last_layer"] = nn.Linear(prev_dim, self._output_dim)
        layers["last_activation"] = self._get_activation(self._last_activation)
        model = nn.Sequential(layers)
        return model
