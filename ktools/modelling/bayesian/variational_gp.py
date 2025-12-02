from typing import Tuple, Union
from numpy import ndarray
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import torch
import gpytorch
import numpy as np
from rich.console import Console
from rich.progress import Progress
from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.models import ApproximateGP
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel
from ktools.modelling.pytorch_utils.set_all_seeds import set_seed
from ktools.modelling.pytorch_utils.numpy_to_tensor_generator import numpy_custom_torch_dataloader


class GPModel(ApproximateGP):

    def __init__(self, 
                 mean_module : Mean,
                 covariance_module : Kernel,
                 inducing_points : torch.Tensor) -> None:
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = mean_module
        self.covar_module = covariance_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalGP(IKtoolsModel):

    def __init__(self,
                 mean_module : Mean,
                 covariance_module : Kernel,
                 noise : Union[torch.Tensor, None] = None,
                 num_inducing_points : int = 500,
                 learning_rate : float = 0.01,
                 training_epochs : int = 10,
                 batch_size : int = 1024,
                 shuffle : bool = True,
                 learn_additional_noise : bool = False,
                 default_noise_level : float = 1e-6,
                 random_state : int = 129,
                 max_kmeans_iter : int = 100,
                 num_inits : int = 3,
                 smart_inducing : bool = True,
                 device_string : str = "cpu") -> None:
        
        self.mean_module = mean_module
        self.covariance_module = covariance_module
        self._noise = noise
        self._num_inducing_points = num_inducing_points
        self._learning_rate = learning_rate
        self._training_epochs = training_epochs
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._default_noise_level = default_noise_level
        self._learn_additional_noise = learn_additional_noise
        self._random_state = random_state
        self._max_kmeans_iter = max_kmeans_iter
        self._num_inits = num_inits
        self._smart_inducing = smart_inducing
        self._device = torch.device(device_string)
        set_seed(random_state)

    def init_model_mll(self, 
                       inducing_points : torch.Tensor, 
                       num_data : int,
                       ) -> Tuple[GPModel, VariationalELBO]:
        """
        Initializes the model and marginal log likelihood (MLL) with the given inducing points.
        """
        model = GPModel(mean_module=self.mean_module,
                        covariance_module=self.covariance_module,
                        inducing_points=inducing_points).to(self._device)
        
        if self._noise is None:
            self.likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(self._default_noise_level/10))
            self.likelihood.noise_covar.initialize(noise=self._default_noise_level)
        else:
            self.likelihood = FixedNoiseGaussianLikelihood(noise=self._noise, learn_additional_noise=self._learn_additional_noise)
        self.likelihood.to(self._device)
        mll = VariationalELBO(self.likelihood, model, num_data=num_data)
        return model, mll
    
    def _find_clusters(self, X : pd.DataFrame, num_clusters : int) -> np.ndarray:
        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            batch_size=self._batch_size,
            max_iter=self._max_kmeans_iter,
            n_init=self._num_inits,
            random_state=self._random_state
        )
        kmeans.fit(X)
        idcs = pd.DataFrame(data={"cluster" : kmeans.labels_}).groupby("cluster", as_index=False).head(1).index
        return idcs

    
    def fit(self,
            X : pd.DataFrame,
            y: pd.DataFrame,
            validation_set: torch.Tuple[ndarray] | None = None,
            val_size: float = 0.05,
            weights: ndarray | None = None) -> IKtoolsModel:
        
        if self._smart_inducing:
            inducing_idcs = self._find_clusters(X, self._num_inducing_points)
            X_inducing = X.iloc[inducing_idcs]
        else:
            X_inducing = X.sample(n=self._num_inducing_points, random_state=self._random_state)
        
        X = X.values
        y = y.values
        num_data = y.shape[0]
        inducing_points = torch.tensor(X_inducing.values, dtype=torch.float32, device=self._device)
        self.model, mll = self.init_model_mll(inducing_points, num_data)

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=self._learning_rate)

        self.model.train()
        self.likelihood.train()
        epochs_iter = range(1, self._training_epochs + 1)
        loss_history = []

        for epoch in epochs_iter:

            total_loss = 0
            train_loader = numpy_custom_torch_dataloader(X, y, batch_size=self._batch_size, shuffle=self._shuffle, dtype=torch.float32)

            # console = Console(force_terminal=True)
            epoch_progress = Progress()
            epoch_progress.start()

            try:
                training = epoch_progress.add_task(f"[red]Training epoch {epoch}...", total=round(num_data / self._batch_size))
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(self._device)
                    y_batch = y_batch.to(self._device)
                    optimizer.zero_grad()
                    output = self.model(x_batch)
                    loss = -mll(output, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    epoch_progress.advance(training, advance=1)
            finally:
                epoch_progress.stop()
            
            loss_history.append(total_loss)
            # print(f"Epoch {epoch + 1}/{self._training_epochs}, Loss: {total_loss}")

        self._loss_history = loss_history
        return self
    
    def predict(self, X: ndarray, return_std : bool = False) -> torch.Tensor:
        X = X.values
        num_data = X.shape[0]
        self.model.eval()
        self.likelihood.eval()
        
        test_loader = numpy_custom_torch_dataloader(X, batch_size=self._batch_size, shuffle=False, dtype=torch.float32)
    
        means = torch.tensor([0.], device=self._device)
        # console = Console(force_terminal=True)
        eval_progress = Progress()
        eval_progress.start()

        try:
            evaluating = eval_progress.add_task(f"[red]Evaluating...", total=round(num_data / self._batch_size))
            with torch.no_grad():
                for (x_batch,) in test_loader:
                    x_batch = x_batch.to(self._device)
                    preds = self.model(x_batch)
                    means = torch.cat([means, preds.mean])
                    eval_progress.advance(evaluating, advance=1)
        finally:
            eval_progress.stop()

        means = means[1:].to("cpu")
        return means.detach().numpy()
    
    @property
    def train_loss(self):
        return self._loss_history