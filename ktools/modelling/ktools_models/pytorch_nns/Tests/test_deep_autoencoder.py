import torch
import torch.nn as nn
import unittest
import logging
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from ktools.modelling.ktools_models.pytorch_nns.deep_autoencoder import DeepAutoencoder
from ktools.modelling.pytorch_utils.pandas_to_tensor_generator import custom_torch_dataloader
from ktools.modelling.pytorch_utils.set_all_seeds import set_seed


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TestDeepAutoencoder(unittest.TestCase):

    def setUp(self):
        set_seed(42)

    def test_deep_autoencoder_forward(self):
        # Arrange
        num_input_features = 64
        levels_of_compression = 3
        batch_size = 64
        expected_compression_size = 8
        input_tensor = torch.randn(batch_size, num_input_features)
        model = DeepAutoencoder(num_input_features, levels_of_compression)

        # Act
        output_tensor = model(input_tensor)
        embedded_tensor = model.encode(input_tensor)

        # Assert
        self.assertEqual(output_tensor.shape, input_tensor.shape)
        self.assertEqual(embedded_tensor.shape, (batch_size, expected_compression_size))

    
    def test_train_deep_autoencoder(self):
        # Arrange
        num_features = 8
        repeats = 8
        levels_of_compression = 2
        X, _ = make_regression(n_samples=1000, n_features=num_features, noise=0.1, random_state=42)
        data = pd.DataFrame(np.repeat(X, repeats, axis=-1), columns=[f'feature_{i}' for i in range(num_features*repeats)])
        expected_final_loss = 8.12897
        
        model = DeepAutoencoder(num_features*repeats, levels_of_compression)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Act
        loss_history = []
        model.train()
        for epoch in range(10):
            batch_dataloader = custom_torch_dataloader(data)
            total_loss = 0
            for (batch,) in batch_dataloader:
                optimiser.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                # print(f"Loss: {loss.item()}")
                loss.backward()
                optimiser.step()

                total_loss += loss.item()     
            loss_history.append(total_loss)
            print(f"Epoch {epoch+1}, Loss: {total_loss}")

        test_array = np.asarray(loss_history)
        loss_difference = test_array[1:] - test_array[:-1]
        self.assertAlmostEqual(expected_final_loss, total_loss, delta=1e-3)
        self.assertTrue((loss_difference < 0).all())
