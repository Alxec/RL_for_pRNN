import random
import numpy as np
import torch
import collections


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(array, signs=False, abs=False):
    d = collections.OrderedDict()
    d["mean"] = np.nanmean(array)
    d["std"] = np.nanstd(array)
    d["min"] = np.nanmin(array)
    d["max"] = np.nanmax(array)
    if signs:
        array = np.array(array)
        d["pos"] = np.sum(np.sign(array)[array>0])
        d["neg"] = np.sum(np.abs(np.sign(array)[array<0]))
        d["ratio"] = d["pos"]/d["neg"] if d["neg"]!=0 else np.nan
    if abs:
        d["abs_mean"] = np.nanmean(np.abs(array))
    return d
