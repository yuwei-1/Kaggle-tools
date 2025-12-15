from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.callbacks import StochasticWeightAveraging
import functools
import torch.nn.functional as F
from torch.nn.modules import Module
from ktools.modelling.ktools_models.pytorch_nns.pytorch_lightning_model import (
    KtoolsBaseLightningmodel,
)
from lifelines.utils import concordance_index
from ktools.modelling.ktools_models.pytorch_nns.embedding_module import (
    EmbeddingCategoricalModule,
)
from pytorch_tabular.models.common.layers import ODST
from ktools.modelling.pytorch_utils.set_all_seeds import set_seed


RANDOM_SEED = 42


@functools.lru_cache
def combinations(N):
    with torch.no_grad():
        ind = torch.arange(N)
        comb = torch.combinations(ind, r=2)
    return comb


def pairwise_loss(
    event: torch.Tensor, event_time: torch.Tensor, risk: torch.Tensor, margin=0.2
):
    n = event.shape[0]
    pairwise_combinations = combinations(n)

    # Find mask
    # first_of_pair, second_of_pair = pairwise_combinations.T
    pairwise_combinations = pairwise_combinations.clone().detach()
    first_of_pair, second_of_pair = (
        pairwise_combinations[:, 0],
        pairwise_combinations[:, 1],
    )
    valid_mask = False
    valid_mask |= (event[first_of_pair] == 1) & (event[second_of_pair] == 1)
    valid_mask |= (event[first_of_pair] == 1) & (
        event_time[first_of_pair] < event_time[second_of_pair]
    )
    valid_mask |= (event[second_of_pair] == 1) & (
        event_time[second_of_pair] < event_time[first_of_pair]
    )

    direction = 2 * (event_time[first_of_pair] > event_time[second_of_pair]).int() - 1
    margin_loss = F.relu(
        -direction * (risk[first_of_pair] - risk[second_of_pair]) + margin
    )

    return (margin_loss.double() * valid_mask.double()).sum() / valid_mask.sum()


class BasicNODE(nn.Module):
    def __init__(self, cat_sizes, emb_sizes, num_numericals) -> None:
        super(BasicNODE, self).__init__()
        self.embedding_layer = EmbeddingCategoricalModule(cat_sizes, emb_sizes)
        emb_dim = self.embedding_layer.concatenated_len
        self.node = nn.Sequential(
            ODST(emb_dim + num_numericals, num_trees=32, tree_output_dim=1),
            # nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_cat, x_num):
        embeddings = self.embedding_layer(x_cat)
        x = torch.cat([embeddings, x_num], dim=1)
        x = self.node(x)
        return x

    def data_aware_init(self, dataloader):
        cats, nums = [], []
        for batch in dataloader:
            x_cat, x_num, *other = batch
            cats += [x_cat]
            nums += [x_num]
        all_cat = torch.cat(cats)
        all_num = torch.cat(nums)

        with torch.no_grad():
            self(all_cat, all_num)


class PostHCTModel(KtoolsBaseLightningmodel):
    def __init__(
        self, model: Module, learning_rate: float, weight_decay: float, race_index: int
    ):
        super().__init__(model, learning_rate, weight_decay)
        self._race_index = race_index

    def get_loss(self, batch, mode=None):
        x_cat, x_num, efs_time, efs = batch
        risk = self(x_cat, x_num).squeeze()

        pwloss = pairwise_loss(efs, efs_time, risk)
        prediction_loss = torch.mean((risk - risk.mean()) ** 2)

        loss_dict = {f"{mode}_pairwise_loss": pwloss}
        batch_dict = {
            "efs_time": [efs_time],
            "efs": [efs],
            "risk_score": [risk.squeeze()],
            "races": [x_cat[:, self._race_index]],
        }
        return pwloss + 0.1 * prediction_loss, loss_dict, batch_dict

    def get_global_metrics(self):
        efs = torch.cat(self.global_metrics["efs"]).cpu().numpy()
        y_hat = torch.cat(self.global_metrics["risk_score"]).cpu().numpy()
        efs_time = torch.cat(self.global_metrics["efs_time"]).cpu().numpy()
        races = torch.cat(self.global_metrics["races"]).cpu().numpy()
        self.global_metrics.clear()

        metric = self._metric(efs, races, efs_time, y_hat)
        cindex = concordance_index(efs_time, y_hat, efs)
        return {
            "stratified concordance index": metric,
            "basic_concordance_index": cindex,
        }

    def _metric(self, efs, races, y, y_hat):
        metric_list = []
        for race in np.unique(races):
            y_ = y[races == race]
            y_hat_ = y_hat[races == race]
            efs_ = efs[races == race]
            metric_list.append(concordance_index(y_, y_hat_, efs_))
        metric = float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
        return metric

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay
        )
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=45, eta_min=6e-3
            ),
            "interval": "epoch",
            "frequency": 1,
            "strict": False,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


def init_ktools_dl(X: pd.DataFrame, y: pd.DataFrame, training=False):
    """
    Initialize data loaders with 4 dimensions : categorical dataframe, numerical dataframe and target values (efs and efs_time).
    Notice that efs_time is log-transformed.
    Fix batch size to 2048 and return dataloader for training or validation depending on training value.
    """

    X_cat = X.select_dtypes("category").values
    X_num = X.select_dtypes("number").values
    ds_train = TensorDataset(
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(y.efs_time.values, dtype=torch.float32).log(),
        torch.tensor(y.efs.values, dtype=torch.long),
        # torch.tensor(quant_y, dtype=torch.float32),
    )
    bs = 2048
    set_seed(RANDOM_SEED)
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=bs, pin_memory=True, shuffle=training
    )
    return dl_train, X_cat, X_num


def train_single_fold(
    train_index,
    test_index,
    X,
    y,
    cat_sizes,
    emb_sizes,
    num_numerics,
    test_df,
    race_idx,
    queue,
):
    print("Running one fold")

    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_val, y_val = X.iloc[test_index], y.iloc[test_index]

    dl_train, X_cat_train, X_num_train = init_ktools_dl(X_train, y_train, training=True)
    dl_val, X_cat_val, X_num_val = init_ktools_dl(X_val, y_val)

    base_model = BasicNODE(cat_sizes, emb_sizes, num_numerics)
    base_model.data_aware_init(deepcopy(dl_train))

    model = PostHCTModel(
        base_model, learning_rate=1e-3, weight_decay=1e-4, race_index=race_idx
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1)
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=60,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
            TQDMProgressBar(),
            StochasticWeightAveraging(
                swa_lrs=1e-5, swa_epoch_start=45, annealing_epochs=15
            ),
        ],
        deterministic=True,
    )
    trainer.fit(model, dl_train)
    trainer.test(model, dl_val)

    oof_pred = model.eval()(
        torch.tensor(X_cat_val, dtype=torch.long),
        torch.tensor(X_num_val, dtype=torch.float32),
    )

    oof_prediction = oof_pred.squeeze().detach().cpu().numpy()

    X_cat_val, X_num_val = (
        test_df.select_dtypes("category").values,
        test_df.select_dtypes("number").values,
    )

    pred = model.eval()(
        torch.tensor(X_cat_val, dtype=torch.long),
        torch.tensor(X_num_val, dtype=torch.float32),
    )
    test_pred = pred.squeeze().detach().cpu().numpy()

    queue.put((oof_prediction, test_pred))
