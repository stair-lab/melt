import random
import numpy as np
import torch


def unique(lst):
    # insert the list to the set
    list_set = set(lst)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list


def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)
