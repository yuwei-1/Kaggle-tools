import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ktools.modelling.ktools_models.pytorch_nns.nonlinear_ff_module import NonLinearFeedForwardModule


class DeepAutoencoder(nn.Module):
    
    def __init__(self,
                 num_input_features : int,
                 levels_of_compression : int):
        self._num_input_features = num_input_features
        self._levels_of_compression = levels_of_compression
        
        compressions = np.cumprod([1] + [2] * levels_of_compression)

        self._encoder = nn.Sequential([
            NonLinearFeedForwardModule(num_input_features // compressions[i],
                                       num_input_features // compressions[i+1],
                                       num_input_features // compressions[i+1])
            for i in range(levels_of_compression - 1)
        ])

        self._decoder = nn.Sequential([
            NonLinearFeedForwardModule(num_input_features // compressions[i],
                                       num_input_features // compressions[i+1],
                                       num_input_features // compressions[i+1])
            for i in range(levels_of_compression - 1)[::-1]
        ])

    def forward(self, x):
        x = self._encoder(x)
        x = self._decoder(x)
        return x