from typing import List
import torch
import torch.nn as nn
from ktools.modelling.Interfaces.i_pytorch_model import IPytorchModel


class BasePytorchEmbeddingModel(IPytorchModel, nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        categorical_idcs: List[int],
        categorical_sizes: List[int],
        categorical_embedding: List[int],
    ) -> None:
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._categorical_idcs = categorical_idcs
        self._categorical_sizes = categorical_sizes
        self._categorical_embedding = categorical_embedding
        self._numerical_idcs = list(
            set(range(input_dim)).difference(set(categorical_idcs))
        )
        self._num_numerics = self._input_dim - self.num_categories
        self._expanded_dim = (
            self._input_dim - self.num_categories + self.combined_emb_dim
        )
        self.embedding_layers = self._create_embedding_layers()
        self.model = self._create_model()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cat, x_num = self.forward_embeddings(x)
        x = torch.cat([x_cat, x_num], dim=1)
        x = self.model(x)
        return x

    def _create_model(self):
        model = nn.Sequential([nn.Linear(self._expanded_dim, self._output_dim)])
        return model

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        cat_inputs = ()
        for i, j in enumerate(self._categorical_idcs):
            feature = x[:, j].long()
            cat_inputs += (self.embedding_layers[i](feature),)

        x_cat = torch.cat(cat_inputs, dim=1)
        x_num = x[:, self._numerical_idcs]
        return x_cat, x_num

    def _create_embedding_layers(self):
        embeddings = []
        for i in range(self.num_categories):
            embeddings += [
                nn.Embedding(self._categorical_sizes[i], self._categorical_embedding[i])
            ]
        return nn.ModuleList(embeddings)

    def _get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "none":
            return nn.Identity()

    @property
    def num_categories(self):
        return len(self._categorical_idcs)

    @property
    def combined_emb_dim(self):
        return sum(self._categorical_embedding)
