import math
import torch
import torch.nn as nn
import pandas as pd
from copy import deepcopy
from typing import List, Any, Callable
from ktools.modelling.ktools_models.pytorch_nns.deep_autoencoder import DeepAutoencoder
from ktools.modelling.pytorch_utils.pandas_to_tensor_generator import custom_torch_dataloader


class DeepFeatureCreator:

    fit = False

    def __init__(self,
                 train_data : pd.DataFrame,
                 features_to_compress : List[str],
                 levels_of_compression : int,
                 max_training_epochs : int = 10,
                 loss : Callable = nn.MSELoss(),
                 learning_rate : float = 1e-3
                 ) -> None:
        self._train_data = train_data
        self._features_to_compress = features_to_compress
        self._levels_of_compression = levels_of_compression
        self._max_training_epochs = max_training_epochs
        self._loss = loss
        self._num_input_features = len(features_to_compress)
        self._autoencoder = DeepAutoencoder(self._num_input_features, levels_of_compression)
        self._optimiser = torch.optim.Adam(self._autoencoder.parameters(), lr=learning_rate)
        self._fit()

    def _fit(self):
        loss_history = []
        best_loss = math.inf
        best_model_weights = None
        self._autoencoder.train()
        for _ in range(self._max_training_epochs):
            batch_dataloader = custom_torch_dataloader(self._train_data)
            total_loss = 0
            for (batch,) in batch_dataloader:
                self._optimiser.zero_grad()
                output = self._autoencoder(batch)
                loss = self._loss(output, batch)
                loss.backward()
                self._optimiser.step()
                total_loss += loss.item()
            
            loss_history += [total_loss]
            if total_loss < best_loss:
                best_model_weights = deepcopy(self._autoencoder.state_dict())
        
        self._autoencoder.load_state_dict(best_model_weights)
        self.fit = True

    def create(self, test_data : pd.DataFrame):
        assert self.fit, "Reintialise object, model not trained successfully."
        self._autoencoder.eval()
        inference_dataloader = custom_torch_dataloader(test_data, shuffle=False)
        encoded_list = []
        for (batch,) in inference_dataloader:
            encoded = self._autoencoder.encode(batch)
            encoded_list += [encoded]
        encoded_test_data = torch.concat(encoded_list)

        dae_test = pd.DataFrame(index=test_data.index, 
                                 data=encoded_test_data.detach().numpy(), 
                                 columns=[f"DAE component {i}" for i in range(encoded_test_data.shape[1])])
        test_data = pd.concat([test_data, dae_test], axis=1)
        return test_data