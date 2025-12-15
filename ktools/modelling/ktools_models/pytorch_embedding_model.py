# from collections import OrderedDict
# import random
# import numpy as np
# from sklearn.model_selection import train_test_split
# from enum import Enum
# import math
# import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import mean_squared_error
# from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel
# from ktools.modelling.ktools_models.pytorch_nns.ffn_pytorch_embedding_model import FFNPytorchEmbeddingModel


# def set_seed(seed: int = 42):
#     random.seed(seed)  # Python random seed
#     np.random.seed(seed)  # NumPy random seed
#     torch.manual_seed(seed)  # PyTorch CPU seed
#     torch.cuda.manual_seed_all(seed)  # PyTorch GPU seed (if available)
#     torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
#     torch.backends.cudnn.benchmark = False  # May impact performance, but ensures reproducibility

# class MyDataset(Dataset):
#     def __init__(self, X, y):
#         X = X.to_numpy().astype(np.float32)
#         y = y.to_numpy().astype(np.float32)
#         self.X = torch.from_numpy(X)
#         self.y = torch.from_numpy(y)

#     def __len__(self):
#         return self.X.shape[0]

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]


# def prep_torch_dataset(X, y, batch_size):
#     torch_dataset = MyDataset(X, y)
#     dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#     return dataloader


# class SupportedPytorchEmbeddingModels(Enum):
#     FEEDFORWARD = FFNPytorchEmbeddingModel
#     ODST = ODSTPytorchEmbeddingModel


# class PytorchEmbeddingModel(IKtoolsModel):

#     def __init__(self,
#                  model_string : str,
#                  input_dim : int,
#                  output_dim : int,
#                  categorical_idcs : List[int],
#                  categorical_sizes : List[int],
#                  categorical_embedding : List[int],
#                  *model_args,
#                  loss=nn.MSELoss(),
#                  metric_callable=mean_squared_error,
#                  optimizer=torch.optim.Adam,
#                  batch_size=64,
#                  epochs=5,
#                  learning_rate=0.001,
#                  decay_period=3,
#                  decay_rate=0.1,
#                  maximise=False,
#                  patience=2,
#                  random_state : int = 42,
#                  verbose : int = 1,
#                  **model_kwargs) -> None:

#         set_seed(random_state)
#         model_cls = SupportedPytorchEmbeddingModels[model_string.upper()].value
#         self.model = model_cls(input_dim,
#                                 output_dim,
#                                 categorical_idcs,
#                                 categorical_sizes,
#                                 categorical_embedding,
#                                 *model_args,
#                                 **model_kwargs)
#         self.initialize_weights()
#         self._loss = loss
#         self._metric_callable = metric_callable
#         self._optimizer = optimizer
#         self._batch_size = batch_size
#         self._epochs = epochs
#         self._learning_rate = learning_rate
#         self._decay_period = decay_period
#         self._decay_rate = decay_rate
#         self._maximise = int(maximise)*2 - 1
#         self._patience = patience
#         self._random_state = random_state
#         self._verbose = verbose

#     def initialize_weights(self):
#         for module in self.model.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)

#     def fit(self, X, y, validation_set = None, val_size=0.05):

#         if validation_set is None:
#             X_train, X_valid, y_train, y_valid = train_test_split(X,
#                                                                   y,
#                                                                   test_size=val_size,
#                                                                   random_state=self._random_state)
#         else:
#             X_train, y_train = X, y
#             X_valid, y_valid = validation_set

#         optimizer = self._optimizer(self.model.parameters(), lr=self._learning_rate)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self._decay_period, gamma=self._decay_rate)
#         dataloader = prep_torch_dataset(X_train, y_train, self._batch_size)
#         model_state_dicts = {}
#         best_epoch = 0
#         best_score = -math.inf

#         for epoch in range(self._epochs):
#             self.model.train()
#             cum_loss = 0
#             nb = 0
#             for (batch_features, batch_target) in dataloader:
#                 nb+=1
#                 optimizer.zero_grad()
#                 predictions = self.model(batch_features)
#                 loss = self._loss(predictions, batch_target)
#                 cum_loss += loss.item()
#                 loss.backward()
#                 optimizer.step()

#             scheduler.step()
#             if self._verbose > 0:
#                 current_lr = scheduler.get_last_lr()[0]
#                 print("Current learning rate: ", current_lr)
#                 print(f"Loss at epoch {epoch + 1}: {cum_loss}")

#             value = self._metric_callable(y_valid, self.predict(X_valid))
#             print(f"{str(self._metric_callable)} value for valid set: {value}")

#             if best_score < self._maximise * value:
#                 best_score = self._maximise * value
#                 best_epoch = epoch
#                 model_state_dicts[epoch] = copy.deepcopy(self.model.state_dict())

#             if (epoch >= best_epoch + self._patience) or (epoch == self._epochs-1):
#                 print(f"Restoring weights at epoch {best_epoch}, score: {self._maximise * best_score}")
#                 self.model.load_state_dict(model_state_dicts[best_epoch])
#                 break

#         return self

#     def _to_tensor(self, X):
#         X = torch.tensor(X.to_numpy(), dtype=torch.float32)
#         return X

#     def predict(self, X):
#         self.model.eval()
#         X = self._to_tensor(X)
#         y_pred = self.model(X).squeeze().detach().numpy()
#         return y_pred
