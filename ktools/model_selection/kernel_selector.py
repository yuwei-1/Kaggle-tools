import torch
from typing import List, Union
from dataclasses import dataclass
from gpytorch.means import Mean
from gpytorch.kernels import Kernel
import matplotlib.pyplot as plt
from ktools.modelling.bayesian.variational_gp import VariationalGP


@dataclass
class GPConfig:
    mean_module: Mean
    covariance_module: Kernel
    name: str = ""
    noise_level: float = 1e-6
    random_state: int = 129


class VariantionalKernelSelector:
    def __init__(
        self,
        configs: List[GPConfig],
        inducing_points: torch.Tensor,
        num_data: int,
        plot_save_path: Union[str, None] = None,
        **gp_kwargs,
    ) -> None:
        self.configs = configs
        self._inducing_points = inducing_points
        self._num_data = num_data
        self._plot_save_path = plot_save_path
        self._gp_kwargs = gp_kwargs

    def select(self, X: torch.Tensor, y: torch.Tensor) -> None:
        names = []
        mlls = []

        for config in self.configs:
            mean_module = config.mean_module
            covariance_module = config.covariance_module
            name = config.name

            gp_model = VariationalGP(
                mean_module,
                covariance_module,
                default_noise_level=config.noise_level,
                random_state=config.random_state,
                **self._gp_kwargs,
            )

            # _, mll = gp_model.init_model_mll(inducing_points=self._inducing_points,
            #                                  num_data=self._num_data)

            gp_model.fit(X, y)
            final_loss = gp_model.train_loss[-1]

            names.append(name)
            mlls.append(final_loss)

        self._plot_performance_of_configs(names, mlls)

    def _plot_performance_of_configs(self, names: List[str], mlls: List[float]) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = range(len(names))

        ax.barh(y_pos, mlls)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=12)
        ax.set_xlabel("Initial Variational ELBO", fontsize=12)
        ax.set_title("Initial ELBO by Kernel Configuration", fontsize=14)
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        if self._plot_save_path:
            plt.savefig(self._plot_save_path)
        plt.show()
