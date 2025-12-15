import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
import sys
import os


# sys.path.append('..')
sys.path.append("/Users/yuwei-1/Documents/projects/Kaggle-tools")
os.chdir("/Users/yuwei-1/Documents/projects/Kaggle-tools")
from post_HCT_survival_notebooks.custom_modules import train_single_fold

from functools import partial, reduce
from ktools.preprocessing.basic_feature_transformers import *
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings
from post_HCT_survival_notebooks.hct_utils import score
import functools
import torch.nn.functional as F


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


def race_equality_loss(race, event, event_time, risk, margin=0.2):
    unq_races, race_counts = torch.unique(race, return_counts=True)
    race_specific_loss = torch.zeros(len(unq_races), dtype=torch.double).to(race.device)
    for i, r in enumerate(unq_races):
        idcs = race == r
        race_specific_loss[i] = pairwise_loss(
            event[idcs], event_time[idcs], risk[idcs], margin=margin
        )
    return torch.std(race_specific_loss)


def scci_metric(
    y_test,
    y_pred,
    id_col_name: str = "ID",
    survived_col_name: str = "efs",
    survival_time_col_name: str = "efs_time",
    stratify_col_name: str = "race_group",
):
    idcs = y_test.index
    og_train = pd.read_csv(train_csv_path)

    y_true = og_train.loc[
        idcs,
        [id_col_name, survived_col_name, survival_time_col_name, stratify_col_name],
    ].copy()
    y_pred_df = og_train.loc[idcs, [id_col_name]].copy()
    y_pred_df["prediction"] = y_pred
    scci = score(y_true.copy(), y_pred_df.copy(), id_col_name)
    return scci


def get_cats():
    df = pd.read_csv(train_csv_path)
    cats = [
        col
        for col in df.columns
        if (2 < df[col].nunique() < 25) | (df[col].dtype == "object")
    ]
    return cats


if __name__ == "__main__":
    ORIGINAL_DATA = False

    train_csv_path = "data/post_hct_survival/train.csv"
    test_csv_path = "data/post_hct_survival/test.csv"
    sub_csv_path = "data/post_hct_survival/sample_submission.csv"
    target_col_name = ["efs", "efs_time"]

    categoricals = get_cats()

    settings = DataSciencePipelineSettings(
        train_csv_path,
        test_csv_path,
        target_col_name,
        categorical_col_names=categoricals,
    )
    transforms = [
        # AddHCTFeatures.transform,
        ImputeNumericalAddIndicator.transform,
        StandardScaleNumerical.transform,
        FillNullValues.transform,
        OrdinalEncode.transform,
        ConvertObjectToCategorical.transform,
        # AddOOFFeatures.transform
    ]

    settings = reduce(lambda acc, func: func(acc), transforms, settings)
    settings.update()

    train, test_df = settings.update()
    test_df.drop(columns=target_col_name, inplace=True)
    X, y = train.drop(columns=settings.target_col_name), train[settings.target_col_name]

    cat_names = settings.categorical_col_names
    cat_sizes = [int(x) for x in X[cat_names].nunique().values]
    emb_sizes = [16] * len(cat_sizes)
    categorical_idcs = [X.columns.get_loc(col) for col in cat_names]
    numerical_idcs = list(set(range(X.shape[1])).difference(set(categorical_idcs)))
    race_idx = X[cat_names].columns.get_loc("race_group")

    import multiprocessing as mp

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    folds = kf.split(X, X.race_group.astype(str))

    if __name__ == "__main__":
        mp.set_start_method("spawn", force=True)
        processes = []
        num_processes = 5
        queue = mp.Queue()

        for train_index, test_index in folds:
            p = mp.Process(
                target=partial(train_single_fold),
                args=(
                    train_index,
                    test_index,
                    X,
                    y,
                    cat_sizes,
                    emb_sizes,
                    len(numerical_idcs),
                    test_df,
                    race_idx,
                    queue,
                ),
            )
            processes.append(p)
            p.start()

        # Wait for all processes to complete.
        for p in processes:
            p.join()

        results = [queue.get() for _ in range(num_processes)]
