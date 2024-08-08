import re
import regex
import numpy as np
from typing import Dict, List
from .utils import normalize_text
from scipy.special import softmax
import ast
from types import SimpleNamespace


def get_json_from_text(text: str, key_answer=None) -> Dict:
    pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")
    jsonObject = pattern.findall(text)

    try:
        processedText = jsonObject[0].replace("\n", "\\n")
        jsonObjectDone = ast.literal_eval(rf"{processedText}")
    except Exception:
        jsonObjectDone = {}
    return jsonObjectDone


def get_class_name_from_text(text: str, class_names: List[str]) -> str:
    text = normalize_text(text)
    class_names = [normalize_text(str(name)) for name in class_names]
    matches = [
        re.search(rf"\b(?:{class_name})\b", text) for class_name in class_names
    ]
    indexes = [match.start() if match else np.inf for match in matches]

    return (
        str(class_names[np.array(indexes).argmin()])
        if min(np.array(indexes)) < np.inf
        else "none"
    )


def softmax_options_prob(options_prob: List):
    options_prob = np.array(options_prob).reshape(len(options_prob), -1)
    return softmax(options_prob, axis=1)


def remove_special_character(text: str) -> str:
    return "".join(letter for letter in text if letter.isalnum())


def get_answer_auto_from_text(
    text: str,
    key_answer: str = None,
    class_names: List[str] = None,
    args=SimpleNamespace(),
) -> str:
    if key_answer:
        json_data = get_json_from_text(text, key_answer)
        if (
            json_data
            and isinstance(json_data, dict)
            and key_answer in json_data
            and json_data[key_answer]
            and remove_special_character(str(json_data[key_answer]))
        ):
            text = str(json_data[key_answer])
        # else:
        #     print(text)
        if class_names:
            text = get_class_name_from_text(text, class_names)
        else:
            text = text

    if "math" not in args.filepath:
        text = text.split("\n\n")[0]
        text = normalize_text(text, keep_punc="keep_punc")
    else:
        if "confident_level" in text:
            text = text[: text.index("confident_level")]
        if f'{{ "{key_answer}":' in text:
            text = text[
                text.index(f'{{ "{key_answer}":')
                + len(f'{{ "{key_answer}":'):
            ]
    return text.replace(",", "").replace(".", "")
