import torch
import numpy as np
from sklearn.model_selection import KFold


def numpy_custom_torch_dataloader(*arrays : np.ndarray, 
                                batch_size : int = 64,
                                shuffle : bool = True,
                                dtype : torch.dtype = torch.double,
                                random_state : int = 42):
    
    num_rows = arrays[0].shape[0]
    if not all(arr.shape[0] == num_rows for arr in arrays):
        raise Exception("All arrays must have the same number of rows.")
    
    batches = round(num_rows / batch_size)
    fold_obj = KFold(n_splits=batches, shuffle=False) if not shuffle else KFold(n_splits=batches, shuffle=True, random_state=random_state)

    convert_batch_to_tensor = lambda arr, idcs : torch.tensor(arr[idcs], dtype=dtype)
    for _, batch_idcs in fold_obj.split(arrays[0]):
        yield tuple(convert_batch_to_tensor(arr, batch_idcs) for arr in arrays)