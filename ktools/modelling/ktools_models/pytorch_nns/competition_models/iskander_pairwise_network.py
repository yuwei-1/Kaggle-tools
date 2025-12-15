import torch
import torch.nn as nn
from typing import *
from ktools.modelling.ktools_models.pytorch_nns.embedding_module import (
    EmbeddingCategoricalModule,
)
from ktools.modelling.ktools_models.pytorch_nns.nonlinear_ff_module import (
    NonLinearFeedForwardModule,
)
from pytorch_tabular.models.common.layers import ODST


class IskanderPairwiseNetwork(nn.Module):
    def __init__(
        self,
        category_cardinalities: List[str],
        numerical_size: int,
        embedding_sizes: List[str],
        embedding_projected_dim: int = 112,
        hidden_size: int = 56,
        output_size: int = 1,
        dropout: float = 0.05463240181423116,
    ):
        super(IskanderPairwiseNetwork, self).__init__()
        self.embedding_module = EmbeddingCategoricalModule(
            category_cardinalities, embedding_sizes
        )
        cat_dim = self.embedding_module.concatenated_len

        self.project_embeddings = NonLinearFeedForwardModule(
            cat_dim, embedding_projected_dim, embedding_projected_dim
        )

        self.aux_predictor = NonLinearFeedForwardModule(
            hidden_size, hidden_size // 3, output_size
        )

        self.odst = nn.Sequential(
            nn.Dropout(dropout),
            ODST(embedding_projected_dim + numerical_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
        )
        self.risk_out = nn.Linear(hidden_size, output_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor):
        emb = self.embedding_module(x_cat)
        emb = self.project_embeddings(emb)
        x = torch.cat([emb, x_num], dim=1)
        x = self.odst(x)
        risk = self.risk_out(x)
        efs_time_pred = self.aux_predictor(x)
        return risk, efs_time_pred

    def data_aware_init(self, dataloader):
        cats, nums = [], []
        for batch in dataloader:
            x_cat, x_num, *other = batch
            cats += [x_cat]
            nums += [x_num]
        all_cat = torch.cat(cats)
        all_num = torch.cat(nums)

        with torch.no_grad():
            self(all_cat, all_num)
