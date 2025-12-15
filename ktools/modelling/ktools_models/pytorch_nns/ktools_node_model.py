import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from pytorch_tabular.models.common.layers import ODST


# TODO: write implementation for multilayer DenseODST, carrying over only max_features
# TODO: write a data aware initialization method


class KtoolsNODE(nn.Module):
    def __init__(
        self,
        input_dim,
        num_trees,
        num_layers,
        tree_output_dim=1,
        input_dropout=0.0,
        flatten_output=False,
    ):
        self._input_dropout = input_dropout
        layers = []
        for _ in num_layers:
            layer = ODST(
                input_dim,
                num_trees=num_trees,
                tree_output_dim=tree_output_dim,
                flatten_output=flatten_output,
            )
            layers += [layer]
            input_dim = input_dim + num_trees * tree_output_dim

        self.node = nn.Sequential(*layers)

    def forward(self, x):
        initial_features = x.shape[-1]
        inp = x
        for layer in self.node:
            h = layer(inp)
            inp = torch.cat([inp, h], dim=1)
            inp = F.dropout(inp, self._input_dropout)

        return inp[..., initial_features:].view(*x.shape[:-1], -1)

    def data_aware_initialization(self, dataloader):
        pass
