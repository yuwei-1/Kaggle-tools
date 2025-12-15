import pytorch_lightning as pl
import torch.nn as nn
import torch
from typing import *
from abc import abstractmethod
from collections import defaultdict


class KtoolsBaseLightningmodel(pl.LightningModule):
    """
    Main Model creation and losses definition to fully train the model.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        weight_decay: float,
    ):
        super(KtoolsBaseLightningmodel, self).__init__()
        self.model = model
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self.global_metrics = defaultdict(list)

    def forward(self, x_cat, x_cont):
        return self.model(x_cat, x_cont)

    @abstractmethod
    def get_loss(self, batch, mode: str):
        assert mode in {"train", "valid", "test"}
        pass

    @abstractmethod
    def get_global_metrics(self):
        pass

    def training_step(self, batch, batch_idx):
        total_loss, loss_dict, batch_metrics = self.get_loss(batch, mode="train")
        for k, v in loss_dict.items():
            self.log(k, v, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, loss_dict, batch_metrics = self.get_loss(batch, mode="valid")
        if batch_idx == 0:
            self.global_metrics.update(batch_metrics)
        else:
            for k, v in batch_metrics.items():
                self.global_metrics[k] += v
        for k, v in loss_dict.items():
            self.log(k, v, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        total_loss, loss_dict, batch_metrics = self.get_loss(batch, mode="test")
        if batch_idx == 0:
            self.global_metrics.update(batch_metrics)
        else:
            for k, v in batch_metrics.items():
                self.global_metrics[k] += v
        for k, v in loss_dict.items():
            self.log(k, v, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def on_validation_epoch_end(self) -> None:
        metric_dict = self.get_global_metrics()
        for k, v in metric_dict.items():
            self.log(k, v, on_epoch=True, prog_bar=True, logger=True)

    def on_test_epoch_end(self) -> None:
        metric_dict = self.get_global_metrics()
        for k, v in metric_dict.items():
            self.log(k, v, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay
        )
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=1, eta_min=6e-3
            ),
            "interval": "epoch",
            "frequency": 1,
            "strict": False,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
