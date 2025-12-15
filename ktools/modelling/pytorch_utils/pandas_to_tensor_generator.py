import torch
import pandas as pd
from sklearn.model_selection import KFold


def pandas_custom_torch_dataloader(
    *dataframes: pd.DataFrame,
    batch_size: int = 64,
    shuffle: bool = True,
    random_state: int = 42,
):
    num_rows = dataframes[0].shape[0]
    if not all(df.shape[0] == num_rows for df in dataframes):
        raise Exception("All dataframes must have the same number of rows.")

    batches = round(num_rows / batch_size)
    fold_obj = (
        KFold(n_splits=batches, shuffle=False)
        if not shuffle
        else KFold(n_splits=batches, shuffle=True, random_state=random_state)
    )

    convert_batch_to_tensor = lambda df, idcs: torch.tensor(
        df.iloc[idcs].values, dtype=torch.float32
    )
    for _, batch_idcs in fold_obj.split(dataframes[0]):
        yield tuple(convert_batch_to_tensor(df, batch_idcs) for df in dataframes)
