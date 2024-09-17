"""
This module provides utility functions for handling various tasks such as:

- Generating unique lists.
- Setting random seeds for reproducibility.
- Extracting columns from matrices.
- Reading and saving JSON and CSV files.
- Formatting data for few-shot learning scenarios.

The utilities are commonly used in data preprocessing and machine learning workflows.
"""

import json
import random

import numpy as np
import pandas as pd
import torch


def unique(lst):
    """
    Returns a unique list by removing duplicates and shuffling the result.

    Args:
        lst (list): The input list.

    Returns:
        list: A shuffled list of unique elements.
    """
    list_set = set(lst)
    unique_list = list(list_set)
    random.shuffle(unique_list)
    return unique_list


def set_seed(seed):
    """
    Sets the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)


def column(matrix, i):
    """
    Extracts a column from a 2D matrix.

    Args:
        matrix (list of list): The 2D matrix.
        i (int): The index of the column to extract.

    Returns:
        list: The extracted column.
    """
    return [row[i] for row in matrix]


def read_json(name, batch_size):
    """
    Reads a JSON file and calculates the current batch index.

    Args:
        name (str): The path to the JSON file.
        batch_size (int): The batch size.

    Returns:
        tuple: A tuple containing the data from the JSON file and the current batch index.
    """
    with open(name, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    current_batch_idx = len(data["references"]) // batch_size
    return data, current_batch_idx


def save_to_json(data, name):
    """
    Saves data to a JSON file.

    Args:
        data (dict): The data to save.
        name (str): The name of the output JSON file.
    """
    json_string = json.dumps(data, indent=4, ensure_ascii=False)
    with open(name, "w", encoding="utf-8") as json_file:
        json_file.write(json_string)


def save_to_csv(data, name):
    """
    Saves data to a CSV file.

    Args:
        data (dict or list): The data to save.
        name (str): The name of the output CSV file.
    """
    df = pd.DataFrame(data)
    df.to_csv(name, index=False)


def format_fewshot(recs, query_format="{}", answer_format="{}"):
    """
    Formats records for few-shot learning.

    Args:
        recs (list): A list of records, where each record is a list with [query, context, answer].
        query_format (str): The format string for the query. Defaults to "{}".
        answer_format (str): The format string for the answer. Defaults to "{}".

    Returns:
        list: A list of conversation dictionaries with roles 'user' and 'assistant'.
    """
    conv = []
    for rec in recs:
        content = query_format.format(*rec[:-1])
        conv.append({"role": "user", "content": content})
        conv.append({"role": "assistant", "content": answer_format.format(rec[-1])})

    return conv
