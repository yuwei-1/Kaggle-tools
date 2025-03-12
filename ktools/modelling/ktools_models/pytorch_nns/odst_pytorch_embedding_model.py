from collections import OrderedDict
from typing import List
import torch
import torch.nn as nn
from pytorch_tabular.models.common.layers import ODST
from ktools.modelling.ktools_models.pytorch_nns.base_pytorch_embedding_model import BasePytorchEmbeddingModel


class ODSTPytorchEmbeddingModel(BasePytorchEmbeddingModel):

    def __init__(self,
                 input_dim : int,
                 output_dim : int,
                 categorical_idcs : List[int],
                 categorical_sizes : List[int],
                 categorical_embedding : List[int],
                 last_activation : str,
                 hidden_dim : int = 256,
                 projection_dim : int = 100,
                 dropout : float = 0.,
                 ):

        self._last_activation = last_activation
        self._projection_dim = projection_dim
        self._hidden_dim = hidden_dim
        self._dropout = dropout

        super().__init__(input_dim,
                         output_dim,
                         categorical_idcs,
                         categorical_sizes,
                         categorical_embedding)
        
        self.projection = nn.Sequential(
            nn.Linear(self.combined_emb_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x_cat, x_num = self.forward_embeddings(x)
        x_proj = self.projection(x_cat)

        x = torch.cat([x_proj, x_num], dim=1)
        x = self.model(x)
        return x
    
    def _create_model(self):
        model = nn.Sequential(
            # nn.Linear(self._expanded_dim, self._hidden_dim),
            nn.Dropout(self._dropout),
            ODST(self._projection_dim + self._num_numerics, self._hidden_dim),
            nn.BatchNorm1d(self._hidden_dim),
            nn.Dropout(self._dropout),
            nn.Linear(self._hidden_dim, self._output_dim),
            self._get_activation(self._last_activation)
        )
        return model