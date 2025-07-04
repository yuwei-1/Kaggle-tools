import torch.nn as nn
import numpy as np
from ktools.modelling.ktools_models.pytorch_nns.nonlinear_ff_module import NonLinearFeedForwardModule


class DeepAutoencoder(nn.Module):
    
    def __init__(self,
                 num_input_features : int,
                 levels_of_compression : int):
        
        super(DeepAutoencoder, self).__init__()

        self._num_input_features = num_input_features
        self._levels_of_compression = levels_of_compression
        
        compressions = np.cumprod([1] + [2] * levels_of_compression)

        self._encoder = nn.Sequential(*[
            NonLinearFeedForwardModule(num_input_features // compressions[i],
                                       num_input_features // compressions[i+1],
                                       num_input_features // compressions[i+1])
            for i in range(levels_of_compression)
        ])

        self._decoder = nn.Sequential(*[
            NonLinearFeedForwardModule(num_input_features // compressions[i+1],
                                       num_input_features // compressions[i],
                                       num_input_features // compressions[i])
            for i in range(levels_of_compression)[::-1]
        ])
    
    def encode(self, x):
        return self._encoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self._decoder(x)
        return x