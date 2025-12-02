import math
import torch
import logging
import torch.nn as nn
import pandas as pd
from copy import deepcopy
from typing import List, Any, Callable, Tuple, Union
from ktools.modelling.ktools_models.pytorch_nns.deep_autoencoder import DeepAutoencoder
from ktools.modelling.pytorch_utils.pandas_to_tensor_generator import pandas_custom_torch_dataloader
from ktools.modelling.pytorch_utils.set_all_seeds import set_seed


class DeepFeatureCreator():

    fit = False

    def __init__(self,
                 train_data : pd.DataFrame,
                 features_to_compress : List[str],
                 levels_of_compression : int,
                 max_training_epochs : int = 10,
                 loss : Callable = nn.MSELoss(),
                 learning_rate : float = 1e-3,
                 min_learning_rate : float = 1e-6,
                 dropout_rate : float = 0.0,
                 lr_reduction_factor : float = 0.5,
                 patience : int = 1,
                 max_norm : float = 1.0,
                 logger : Union[None, logging.Logger] = None,
                 random_state : int = 129,
                 device_string : str = "cpu"
                 ) -> None:
        
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

        self.logger.info("Successfully initialised logger.")
        self._train_data = train_data
        self._features_to_compress = features_to_compress
        self._levels_of_compression = levels_of_compression
        self._max_training_epochs = max_training_epochs
        self._loss = loss
        self._num_input_features = len(features_to_compress)
        self._max_norm = max_norm
        self._device = torch.device(device_string)
        set_seed(random_state)

        self._autoencoder = DeepAutoencoder(
            self._num_input_features, 
            levels_of_compression,
            dropout_rate
        ).to(self._device)

        self._optimiser = torch.optim.Adam(self._autoencoder.parameters(), lr=learning_rate)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimiser,
            mode='min',
            factor=lr_reduction_factor,
            patience=patience,
            min_lr=min_learning_rate,
            # verbose=True
        )

        self._loss_history = self._fit()

    def _fit(self):
        loss_history = []
        best_loss = math.inf
        best_model_weights = None
        self._autoencoder.train()
        for epoch in range(1, self._max_training_epochs+1):
            batch_dataloader = pandas_custom_torch_dataloader(self._train_data, random_state=epoch)
            total_loss = 0
            for (batch,) in batch_dataloader:
                batch = batch.to(self._device)
                self._optimiser.zero_grad()
                output = self._autoencoder(batch)
                loss = self._loss(output, batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self._autoencoder.parameters(), max_norm=self._max_norm)
                self._optimiser.step()
                total_loss += loss.item()
            
            self.logger.info(f"The loss for epoch {epoch} is {total_loss}")
            self._scheduler.step(total_loss)
            loss_history += [total_loss]
            if total_loss < best_loss:
                best_loss = total_loss
                best_model_weights = deepcopy(self._autoencoder.state_dict())
        
        self._autoencoder.load_state_dict(best_model_weights)
        self.fit = True
        return loss_history

    def create(self, df : pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        assert self.fit, "Reintialise object, model not trained successfully."
        self._autoencoder.eval()
        inference_dataloader = pandas_custom_torch_dataloader(df, shuffle=False)
        encoded_list = []
        for (batch,) in inference_dataloader:
            batch = batch.to(self._device)
            encoded = self._autoencoder.encode(batch)
            encoded_list += [encoded]
        encoded_df = torch.concat(encoded_list).to("cpu")

        new_col_names = [f"DAE component {i}" for i in range(encoded_df.shape[1])]
        dae_test = pd.DataFrame(index=df.index, 
                                 data=encoded_df.detach().numpy(), 
                                 columns=new_col_names)
        df = pd.concat([df, dae_test], axis=1)
        return df, new_col_names
    
    @property
    def train_loss(self):
        return self._loss_history