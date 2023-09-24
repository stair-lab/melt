import random
import json
import numpy as np
import pandas as pd
import torch


def unique(lst):
    # insert the list to the set
    list_set = set(lst)
    # convert the set to the list
    unique_list = list(list_set)
    return unique_list


def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)

def column(matrix, i):
    return [row[i] for row in matrix]


def save_to_json(data, name):
    jsonString = json.dumps(data, indent=4, ensure_ascii=False)
    jsonFile = open(name, "w")
    jsonFile.write(jsonString)
    jsonFile.close()


def save_to_csv(data, name):
    df = pd.DataFrame(data)
    df.to_csv(name, index=False)
