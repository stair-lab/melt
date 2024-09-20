"post_process"
import re
from typing import Dict, List
import ast
from types import SimpleNamespace
import regex
from scipy.special import softmax
import numpy as np
from melt.tools.metrics.utils import normalize_text

def get_json_from_text(text: str) -> Dict:
    "function"
    pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")
    json_objects = pattern.findall(text)
    try:
        processed_text = json_objects[0].replace("\n", "\\n")
        json_object_result = ast.literal_eval(rf"{processed_text}")
    except (IndexError, SyntaxError, ValueError):
        json_object_result = {}
    return json_object_result
def get_class_name_from_text(text: str, class_names: List[str]) -> str:
    "function"
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
    "function"
    options_prob = np.array(options_prob).reshape(len(options_prob), -1)
    return softmax(options_prob, axis=1)
def remove_special_character(text: str) -> str:
    "function"
    return "".join(letter for letter in text if letter.isalnum())
def get_answer_auto_from_text(
    text: str,
    key_answer: str = None,
    class_names: List[str] = None,
    args=SimpleNamespace(),
) -> str:
    "function"
    if key_answer:
        json_data = get_json_from_text(text)
        if (
            json_data
            and isinstance(json_data, dict)
            and key_answer in json_data
            and json_data[key_answer]
            and remove_special_character(str(json_data[key_answer]))
        ):
            text = str(json_data[key_answer])
        if class_names:
            text = get_class_name_from_text(text, class_names)
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
