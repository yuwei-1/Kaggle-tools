import random
import numpy as np
import torch
import pytorch_lightning as pl

def set_seed(seed=42, device='cpu'):
    if device == 'cpu':
        deter = True
        print("CUDA backend is now deterministic, only CPU is being used")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deter
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)