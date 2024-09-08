"""
This module provides functions for processing and extracting information from text.
"""
import ast
import re
from types import SimpleNamespace
from typing import Dict, List
import numpy as np
from scipy.special import softmax
from .utils import normalize_text

try:
    import regex
except ImportError:
    print("The 'regex' library is not installed. Please install it using 'pip install regex'.")


def get_json_from_text(text: str) -> Dict:
    """Extracts JSON-like objects from text."""
    pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")
    json_objects = pattern.findall(text)

    try:
        if json_objects:
            processed_text = json_objects[0].replace("\n", "\\n")
            json_object_done = ast.literal_eval(processed_text)
        else:
            json_object_done = {}
    except (SyntaxError, ValueError) as e:
        print(f"Error processing JSON: {e}")
        json_object_done = {}
    return json_object_done


def get_class_name_from_text(text: str, class_names: List[str]) -> str:
    """Finds the class name from the text that matches the provided class names."""
    text = normalize_text(text)
    class_names = [normalize_text(name) for name in class_names]
    matches = [
        re.search(rf"\b(?:{class_name})\b", text) for class_name in class_names
    ]
    indexes = [match.start() if match else np.inf for match in matches]

    return (
        class_names[np.array(indexes).argmin()]
        if min(np.array(indexes)) < np.inf
        else "none"
    )


def softmax_options_prob(options_prob: List) -> np.ndarray:
    """Applies softmax to options probabilities."""
    options_prob = np.array(options_prob).reshape(len(options_prob), -1)
    return softmax(options_prob, axis=1)


def remove_special_character(text: str) -> str:
    """Removes non-alphanumeric characters from the text."""
    return "".join(letter for letter in text if letter.isalnum())


def get_answer_auto_from_text(
    text: str,
    key_answer: str = None,
    class_names: List[str] = None,
    args=SimpleNamespace(),
) -> str:
    """Extracts and processes an answer from the text based on the provided arguments."""
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
