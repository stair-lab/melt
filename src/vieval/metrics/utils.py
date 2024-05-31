import json
from typing import Dict
import os
import pandas as pd
from nltk.metrics.scores import f_measure


def normalize_text(text: str, keep_punc=False) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    if keep_punc:
        text = white_space_fix(lower(text))
    else:
        text = white_space_fix(remove_punc(lower(text)))

    if len(text) == 0:
        text = "."

    return text


def read_json_file(filepath: str) -> Dict:
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf8") as f:
            data = json.load(f)
    else:
        data = {}

    return data


def save_to_json(data: Dict, filename: str, outdir: str = "./"):
    filepath = os.path.join(outdir, filename)
    old_data = read_json_file(filepath)
    old_data.update(data)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(old_data, f, ensure_ascii=False, indent=4)


def save_to_csv(data: Dict, filename: str, outdir: str = "./"):
    filepath = os.path.join(outdir, filename)
    data.to_csv(filepath, mode="x")


def save_to_xlsx(data, filename: str):
    writer = pd.ExcelWriter(filename)
    _ = [
        sheet_data.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
        for sheet_name, sheet_data in data.items()
    ]
    writer.close()
