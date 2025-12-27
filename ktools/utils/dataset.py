import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(
        self,
        X_num: pd.DataFrame | np.ndarray,
        X_cat: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
    ):
        self.X_num = X_num.to_numpy() if isinstance(X_num, pd.DataFrame) else X_num
        self.X_cat = X_cat.to_numpy() if isinstance(X_cat, pd.DataFrame) else X_cat

        self.y = y
        if y is not None:
            self.y = (
                y.to_numpy().reshape(-1, 1)
                if isinstance(y, pd.Series)
                else y.reshape(-1, 1)
            )

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        x_num = torch.from_numpy(self.X_num[idx]).to(torch.float32)
        x_cat = torch.from_numpy(self.X_cat[idx]).to(torch.long)
        if self.y is not None:
            y = torch.from_numpy(self.y[idx])
            return x_num, x_cat, y
        return x_num, x_cat
