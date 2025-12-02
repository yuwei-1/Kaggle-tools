import torch.nn as nn
import numpy as np
from ktools.modelling.ktools_models.pytorch_nns.nonlinear_ff_module import NonLinearFeedForwardModule


class DeepAutoencoder(nn.Module):
    
    def __init__(self,
                 num_input_features : int,
                 levels_of_compression : int,
                 dropout_rate : float = 0.):
        
        super(DeepAutoencoder, self).__init__()

        self._num_input_features = num_input_features
        self._levels_of_compression = levels_of_compression
        
        compressions = np.cumprod([1] + [2] * levels_of_compression)

        encoder_modules = []
        for i in range(levels_of_compression):
            encoder_modules.append(
                NonLinearFeedForwardModule(num_input_features // compressions[i],
                                           num_input_features // compressions[i+1],
                                           num_input_features // compressions[i+1])
            )
            if i < levels_of_compression - 1:
                encoder_modules.append(nn.Dropout(p=dropout_rate))
                encoder_modules.append(nn.BatchNorm1d(num_input_features // compressions[i+1]))

        self._encoder = nn.Sequential(*encoder_modules)

        decoder_modules = []
        for i in range(levels_of_compression)[::-1]:
            decoder_modules.append(
                NonLinearFeedForwardModule(num_input_features // compressions[i+1],
                                           num_input_features // compressions[i],
                                           num_input_features // compressions[i])
            )
            if i < levels_of_compression - 1:
                decoder_modules.append(nn.Dropout(p=dropout_rate))
                decoder_modules.append(nn.BatchNorm1d(num_input_features // compressions[i]))

        self._decoder = nn.Sequential(*decoder_modules)

        # self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def encode(self, x):
        return self._encoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self._decoder(x)
        return x