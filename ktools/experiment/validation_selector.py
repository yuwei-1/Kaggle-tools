from typing import Callable
import numpy as np
import pandas as pd
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from sklearn.model_selection import BaseCrossValidator

from ktools.base.model import BaseKtoolsModel
from ktools.config.dataset import DatasetConfig


class ValidationSelector:
    def __init__(
        self,
        model: BaseKtoolsModel,
        outer_fold_splitter: BaseCrossValidator,
        inner_fold_splitter: BaseCrossValidator,
        metric: Callable,
    ):
        self.outer_fold_splitter = outer_fold_splitter
        self.inner_fold_splitter = inner_fold_splitter
        self.model = model
        self.metric = metric

    def run(self, train_data: pd.DataFrame, config: DatasetConfig):
        fold_results = {}
        n_outer_splits = self.outer_fold_splitter.get_n_splits()
        n_inner_splits = self.inner_fold_splitter.get_n_splits()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[bold green]{task.fields[status]}"),
        ) as progress:
            outer_task = progress.add_task(
                "Outer Folds", total=n_outer_splits, status=""
            )

            for outer_fold_idx, (outer_train_idcs, outer_test_idcs) in enumerate(
                self.outer_fold_splitter.split(
                    train_data,
                    train_data[config.target_col_name],
                )
            ):
                progress.update(
                    outer_task, status=f"Fold {outer_fold_idx + 1}/{n_outer_splits}"
                )
                outer_train_data = train_data.iloc[outer_train_idcs]
                outer_test_data = train_data.iloc[outer_test_idcs]

                inner_cv_scores, simulated_test_set_score = self._run_inner_cv(
                    outer_train_data,
                    outer_test_data,
                    config,
                    progress,
                    n_inner_splits,
                )

                fold_results[f"fold_{outer_fold_idx}"] = {
                    "inner_cv_scores": inner_cv_scores,
                    "simulated_test_set_score": simulated_test_set_score,
                }
                progress.advance(outer_task)

        return fold_results

    def _run_inner_cv(
        self,
        inner_train_data: pd.DataFrame,
        inner_val_data: pd.DataFrame,
        config: DatasetConfig,
        progress: Progress,
        n_inner_splits: int,
    ):
        scores = []
        simulated_test_set_preds = np.empty((inner_val_data.shape[0],))

        inner_task = progress.add_task("  Inner Folds", total=n_inner_splits, status="")

        for inner_fold_idx, (inner_train_idcs, inner_val_idcs) in enumerate(
            self.inner_fold_splitter.split(
                inner_train_data,
                inner_train_data[config.target_col_name],
            )
        ):
            progress.update(
                inner_task, status=f"Fold {inner_fold_idx + 1}/{n_inner_splits}"
            )
            inner_train_fold = inner_train_data.iloc[inner_train_idcs]
            inner_val_fold = inner_train_data.iloc[inner_val_idcs]

            self.model.fit(
                X=inner_train_fold[config.training_col_names],
                y=inner_train_fold[config.target_col_name],
                validation_set=(
                    inner_val_fold[config.training_col_names],
                    inner_val_fold[config.target_col_name],
                ),
            )

            val_preds = self.model.predict(inner_val_fold[config.training_col_names])
            test_preds = self.model.predict(inner_val_data[config.training_col_names])
            score = self.metric(
                inner_val_fold[config.target_col_name],
                val_preds,
            )
            scores.append(score)
            simulated_test_set_preds += (
                test_preds / self.inner_fold_splitter.get_n_splits()
            )
            progress.advance(inner_task)

        progress.remove_task(inner_task)

        simulated_test_set_score = self.metric(
            inner_val_data[config.target_col_name],
            simulated_test_set_preds,
        )

        return scores, simulated_test_set_score
