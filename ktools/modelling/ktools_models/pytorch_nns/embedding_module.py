import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *



class EmbeddingCategoricalModule(nn.Module):
    """
    Embed categorical feature
    """

    def __init__(self,
                 category_cardinalities : List[str],
                 embedding_sizes : List[str],
                 projection_dim : Union[None, int] = None) -> None:
        
        super(EmbeddingCategoricalModule, self).__init__()
        self._category_cardinalities = category_cardinalities
        self._embedding_sizes = embedding_sizes
        self._embedding_layers = self._build_embedding_layers()

        if projection_dim is not None:
            self.mlp = nn.Linear(self.concatenated_len, projection_dim)
        else:
            self.mlp = nn.Identity()

    def forward(self, x_cat : torch.Tensor) -> torch.Tensor:
        x = [embedder(x_cat[:, i]) for i, embedder in enumerate(self._embedding_layers)]
        x = torch.cat(x, dim=1)
        x = self.mlp(x)
        return x

    @property
    def num_features(self):
        return len(self._embedding_sizes)
    
    @property
    def concatenated_len(self):
        return sum(self._embedding_sizes)

    def _build_embedding_layers(self):
        embedding_layers = nn.ModuleList([
            nn.Embedding(self._category_cardinalities[i], self._embedding_sizes[i]) for i in range(self.num_features)
            ])
        return embedding_layers