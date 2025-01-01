from collections import OrderedDict
import numpy as np
from sklearn.model_selection import train_test_split
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from ktools.modelling.Interfaces.i_sklearn_model import ISklearnModel



class BasicFeedForwardNetwork(nn.Module):

    def __init__(self,
                 input_dim : int,
                 output_dim : int,
                 categorical_idcs : List[int],
                 categorical_sizes : List[int],
                 categorical_embedding : List[int],
                 activation : str,
                 last_activation : str,
                 num_hidden_layers : int = 1,
                 largest_hidden_dim :int = 256,
                 dim_decay : float = 1.0,
                 ):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        
        self._categorical_idcs = categorical_idcs
        self._categorical_sizes = categorical_sizes
        self._categorical_embedding = categorical_embedding
        self._num_categories = len(categorical_idcs)
        self._activation = activation
        self._last_activation = last_activation

        self._expanded_dim = self._input_dim - self._num_categories + sum(self._categorical_embedding)
        self._largest_hidden_dim = largest_hidden_dim
        self._num_hidden_layers = num_hidden_layers
        self._dim_decay = dim_decay
        
        self.embedding_layers = self._create_embedding_layers()
        self.model = self._create_dense_layers()

    def forward(self, x) -> torch.Tensor:
        x = self.forward_embeddings(x)
        x = self.model(x)
        return x
      
    def forward_embeddings(self, x):
        inputs = ()
        for i in range(self._input_dim):
            if i in self._categorical_idcs:
                feature = x[:, i].long()
            else:
                feature = x[:, i:i+1]
            inputs += (self.embedding_layers[i](feature),)
        x = torch.cat(inputs, dim=1)
        return x
    
    def _create_dense_layers(self):
        layers = OrderedDict()
        prev_dim = self._expanded_dim
        curr_dim = self._largest_hidden_dim

        for l in range(self._num_hidden_layers):
            layers[f'layer_{l}'] = nn.Linear(prev_dim, curr_dim)
            layers[f'activation_{l}'] = self._get_activation(self._activation)
            prev_dim = curr_dim
            curr_dim = max(int(curr_dim*self._dim_decay), self._output_dim)
        
        layers['last_layer'] = nn.Linear(prev_dim, self._output_dim)
        layers['last_activation'] = self._get_activation(self._last_activation)
        model = nn.Sequential(layers)
        return model

    def _create_embedding_layers(self):
        embeddings = []
        for i in range(self._input_dim):
            if i in self._categorical_idcs:
                j = self._categorical_idcs.index(i)
                embeddings += [nn.Embedding(self._categorical_sizes[j], self._categorical_embedding[j])]
            else:
                embeddings += [nn.Identity()]
        return embeddings
    
    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'none':
            return nn.Identity()
        

class MyDataset(Dataset):
    def __init__(self, X, y):
        # self.X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        # self.y = torch.tensor(y.to_numpy(), dtype=torch.float32)
        X = X.to_numpy().astype(np.float32)
        y = y.to_numpy().astype(np.float32)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

        # self.X = torch.from_numpy(X.to_numpy())
        # self.y = torch.from_numpy(y.to_numpy())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def prep_torch_dataset(X, y, batch_size):
    torch_dataset = MyDataset(X, y)
    dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
    # dataloader = CustomDataLoader(X, y, batch_size).split_data()
    return dataloader


class PytorchFFNModel(ISklearnModel):
    
    def __init__(self,
                 input_dim : int,
                 output_dim : int,
                 categorical_idcs : List[int],
                 categorical_sizes : List[int],
                 categorical_embedding : List[int],
                 activation : str,
                 last_activation : str,
                 num_hidden_layers : int = 1,
                 largest_hidden_dim :int = 256,
                 dim_decay : float = 1.0,
                 loss=nn.MSELoss(),
                 metric_callable=mean_squared_error,
                 optimizer=torch.optim.Adam,
                 batch_size=64,
                 epochs=5,
                 learning_rate=0.001,
                 decay_period=3,
                 decay_rate=0.1,
                 maximise=False,
                 patience=2,
                 random_state : int = 42,
                 verbose : int = 1) -> None:
        
        torch.manual_seed(random_state)
        self.model = BasicFeedForwardNetwork(input_dim,
                                            output_dim,
                                            categorical_idcs,
                                            categorical_sizes,
                                            categorical_embedding,
                                            activation,
                                            last_activation,
                                            num_hidden_layers,
                                            largest_hidden_dim,
                                            dim_decay)
        self.initialize_weights()
        self._loss = loss
        self._metric_callable = metric_callable
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._decay_period = decay_period
        self._decay_rate = decay_rate
        self._maximise = int(maximise)*2 - 1
        self._patience = patience
        self._random_state = random_state
        self._verbose = verbose
        
    def initialize_weights(self):
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def fit(self, X, y, validation_set = None, val_size=0.05):

        if validation_set is None:
            X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                                  y, 
                                                                  test_size=val_size, 
                                                                  random_state=self._random_state)
        else:
            X_train, y_train = X, y
            X_valid, y_valid = validation_set

        optimizer = self._optimizer(self.model.parameters(), lr=self._learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self._decay_period, gamma=self._decay_rate)
        dataloader = prep_torch_dataset(X_train, y_train, self._batch_size)
        model_state_dicts = {}
        best_epoch = 0
        best_score = -math.inf

        for epoch in range(self._epochs):
            self.model.train()
            cum_loss = 0
            nb = 0
            for (batch_features, batch_target) in dataloader:
                nb+=1
                optimizer.zero_grad()
                predictions = self.model(batch_features)
                loss = self._loss(predictions, batch_target)
                cum_loss += loss.item()
                loss.backward()
                optimizer.step()

            scheduler.step()
            if self._verbose > 0:
                current_lr = scheduler.get_last_lr()[0]
                print("Current learning rate: ", current_lr)
                print(f"Loss at epoch {epoch + 1}: {cum_loss}")

            value = self._metric_callable(y_valid, self.predict(X_valid))
            print(f"{str(self._metric_callable)} value for valid set: {value}")

            if best_score < self._maximise * value:
                print(f"here at epoch {epoch}")
                best_score = self._maximise * value
                best_epoch = epoch
                model_state_dicts[epoch] = copy.deepcopy(self.model.state_dict())
                
            if (epoch >= best_epoch + self._patience) or (epoch == self._epochs-1):
                print(f"Restoring weights at epoch {best_epoch}, score: {self._maximise * best_score}")
                self.model.load_state_dict(model_state_dicts[best_epoch])
                break

        return self
        
    def _to_tensor(self, X):
        X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        return X

    def predict(self, X):
        self.model.eval()
        X = self._to_tensor(X)
        y_pred = self.model(X).squeeze().detach().numpy()
        return y_pred