import json
import random

import numpy as np
import pandas as pd
import torch

from urllib.parse import urlparse
from os.path import exists

def is_local(url):
    url_parsed = urlparse(url)
    if url_parsed.scheme in ('file', ''): # Possibly a local file
        return exists(url_parsed.path)
    return False
    
def unique(lst):
    # insert the list to the set
    list_set = set(lst)
    # convert the set to the list
    unique_list = list(list_set)
    random.shuffle(unique_list)
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


def read_json(name, batch_size):
    with open(name, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    current_batch_idx = len(data["references"]) // batch_size
    # if 'fewshot' in data:
    #     fewshot = data['fewshot']
    # else:
    #     fewshot = None
    # try:
    #     del data['fewshot']
    # except:
    #     pass
    # df = pd.DataFrame(data)
    return data, current_batch_idx


def save_to_json(data, name):
    jsonString = json.dumps(data, indent=4, ensure_ascii=False)
    jsonFile = open(name, "w", encoding="utf-8")
    jsonFile.write(jsonString)
    jsonFile.close()


def save_to_csv(data, name):
    df = pd.DataFrame(data)
    df.to_csv(name, index=False)


def format_fewshot(recs, query_format="{}", answer_format="{}"):
    conv = []
    for rec in recs:  # rec: [sampling_rate, query, context, answer]
        if "[modal]" in query_format:
            tmp_content = query_format.format(*rec[1:-1])
            content = []
            tmp_content = tmp_content.split("[modal]")
            content = [
                {"type": "text", "text": tmp_content[0]},
                {"type": "audio", "audio_url": rec[0]},
                {"type": "text", "text": tmp_content[1]},
            ]
        else:
            content = query_format.format(*rec[:-1])
        conv.append({"role": "user", "content": content})
        conv.append(
            {"role": "assistant", "content": answer_format.format(rec[-1])}
        )

    return conv
