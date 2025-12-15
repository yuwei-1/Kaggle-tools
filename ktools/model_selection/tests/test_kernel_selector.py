import unittest
import torch
import pandas as pd
from gpytorch.means import ConstantMean
from gpytorch.kernels import *
from ktools.model_selection.kernel_selector import VariantionalKernelSelector, GPConfig


class TestKernelSelector(unittest.TestCase):
    def setUp(self):
        train_size = 785

        self.configs = [
            # 1) Recommended starting kernel from docs
            GPConfig(
                mean_module=ConstantMean().initialize(constant=0.0),
                covariance_module=ScaleKernel(RBFKernel(ard_num_dims=train_size))
                + ConstantKernel(),
                name="ScaleKernel(RBFKernel()) + ConstantKernel()",
            ),
            GPConfig(
                mean_module=ConstantMean().initialize(constant=0.0),
                covariance_module=ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=train_size)
                )
                + ConstantKernel(),
                name="ScaleKernel(MaternKernel(nu=2.5)) + ConstantKernel()",
            ),
            GPConfig(
                mean_module=ConstantMean().initialize(constant=0.0),
                covariance_module=ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=train_size)
                ),
                name="ScaleKernel(MaternKernel(nu=2.5))",
            ),
            GPConfig(
                mean_module=ConstantMean().initialize(constant=0.0),
                covariance_module=ScaleKernel(RBFKernel(ard_num_dims=train_size)),
                name="ScaleKernel(RBFKernel(ard_num_dims=train_size))",
            ),
            # GPConfig(
            #     mean_module=ConstantMean().initialize(constant=0.0),
            #     covariance_module=ScaleKernel(RBFKernel(ard_num_dims=train_size)) + ConstantKernel(),
            #     name="ScaleKernel(RBFKernel(ard_num_dims=train_size))"
            # ),
            # GPConfig(
            #     mean_module=ConstantMean().initialize(constant=0.0),
            #     covariance_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_size)) + ConstantKernel(),
            #     name="ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_size)) + ConstantKernel()"
            # ),
            # GPConfig(
            #     mean_module=ConstantMean().initialize(constant=0.0),
            #     covariance_module=ScaleKernel(RQKernel(ard_num_dims=train_size)) + ConstantKernel(),
            #     name="ScaleKernel(RQKernel(ard_num_dims=train_size)) + ConstantKernel()"
            # ),
            GPConfig(
                mean_module=ConstantMean().initialize(constant=0.0),
                covariance_module=ScaleKernel(RQKernel(ard_num_dims=train_size))
                + ConstantKernel(),
                name="ScaleKernel(RQKernel(ard_num_dims=train_size)) + ConstantKernel()",
            ),
            GPConfig(
                mean_module=ConstantMean().initialize(constant=0.0),
                covariance_module=ScaleKernel(RQKernel(ard_num_dims=train_size)),
                name="ScaleKernel(RQKernel(ard_num_dims=train_size))",
            ),
            # 4) Spectral Mixture (4 components) + zero mean
            # GPConfig(
            #     mean_module=ConstantMean().initialize(constant=0.0),
            #     covariance_module=ScaleKernel(SpectralMixtureKernel(num_mixtures=2, ard_num_dims=train_size))
            # )
        ]

    def test_select(self):
        # Arrange
        X = pd.read_csv(
            "./ktools/model_selection/tests/test_data/test_sample_data.csv", index_col=0
        )
        y = X.pop("label")
        subset_X = X.sample(n=1000)

        selector = VariantionalKernelSelector(
            self.configs,
            inducing_points=torch.tensor(subset_X.values, dtype=torch.float32),
            num_data=X.shape[0],
            plot_save_path="./ktools/model_selection/tests/test_data/mll_plot.png",
            training_epochs=10,
        )

        # Act
        selector.select(X, y)
