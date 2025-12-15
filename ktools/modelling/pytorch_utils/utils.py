import torch.nn as nn


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "none":
        return nn.Identity()
