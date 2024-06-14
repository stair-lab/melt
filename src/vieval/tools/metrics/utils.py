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
