import json
import torch


def convert2torch(*args):
    res = tuple([torch.tensor(arg, dtype=torch.int64) for arg in args])
    return res