"""
This module contains the ReasoningMetric class, which evaluates the performance
of a reasoning task by calculating F1 scores, exact match scores, and equality scores
between predictions and references. It includes functions to handle mathematical
expressions and formatting.

The ReasoningMetric class inherits from the BaseMetric class and implements the
evaluate method to compute these metrics.
"""

from typing import Dict
import numpy as np
from .basic_metrics import exact_match, f1_score
from .base import BaseMetric

escape_dict = {
    "\a": r"\a",
    "\b": r"\b",
    "\f": r"\f",
    "\n": r"\n",
    "\r": r"\r",
    "\t": r"\t",
    "\v": r"\v",
}


def _fix_fracs(string: str) -> str:
    """
    Fixes fractions in the given string by ensuring proper formatting.

    Args:
        string (str): The input string potentially containing fractions.

    Returns:
        str: The formatted string with corrected fractions.
    """
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += f"{{{a}}}{{{b}}}{post_substr}"
                    else:
                        new_str += f"{{{a}}}{{{b}}}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += f"{{{a}}}{b}{post_substr}"
                    else:
                        new_str += f"{{{a}}}{b}"
    return new_str


def _fix_a_slash_b(string: str) -> str:
    """
    Converts a simple fraction in the form of 'a/b' into LaTeX format.

    Args:
        string (str): The input string potentially containing a fraction.

    Returns:
        str: The LaTeX formatted fraction.
    """
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a = int(a)
        b = int(b)
        assert string == f"{a}/{b}"
        return f"\\frac{{{a}}}{{{b}}}"
    except (ValueError, AssertionError):
        return string


def _remove_right_units(string: str) -> str:
    """
    Removes units from the right side of the string.

    Args:
        string (str): The input string potentially containing units.

    Returns:
        str: The string with units removed.
    """
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    """
    Fixes square roots in the given string by ensuring proper formatting.

    Args:
        string (str): The input string potentially containing square roots.

    Returns:
        str: The formatted string with corrected square roots.
    """
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = f"\\sqrt{{{a}}}{split[1:]}"
        else:
            new_substr = f"\\sqrt{split}"
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    """
    Cleans and formats the input string by removing unnecessary characters and formatting.

    Args:
        string (str): The input string to be cleaned.

    Returns:
        str: The cleaned and formatted string.
    """
    # Line breaks
    string = string.replace("\n", "")

    # Remove inverse spaces
    string = string.replace("\\!", "")

    # Replace \\ with \
    string = string.replace("\\\\", "\\")

    # Replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # Remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # Remove dollar signs
    string = string.replace("\\$", "")

    # Remove units (on the right)
    string = _remove_right_units(string)

    # Remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{."
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = f"0{string}"

    # Remove "X = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # Fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # Remove spaces
    string = string.replace(" ", "")

    # Fix fractions
    string = _fix_fracs(string)

    # Change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # Fix simple fractions
    string = _fix_a_slash_b(string)

    return string


def is_equiv(str1: str, str2: str, verbose=False) -> bool:
    """
    Checks if two strings are equivalent after formatting.

    Args:
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.
        verbose (bool): If True, prints the formatted strings.

    Returns:
        bool: True if the strings are equivalent, False otherwise.
    """
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except ValueError:
        return str1 == str2


class ReasoningMetric(BaseMetric):
    """Metric for evaluating reasoning tasks, including mathematical expressions."""

    def equal(self, prediction: str, reference: str) -> float:
        """
        Checks if a prediction is equal to the reference.

        Args:
            prediction (str): The predicted string.
            reference (str): The reference string.

        Returns:
            float: 1 if equal, 0 otherwise.
        """
        if prediction == reference:
            return 1
        return 0

    def _has_numbers(self, word: str) -> bool:
        """
        Checks if a word contains any digits.

        Args:
            word (str): The word to check.

        Returns:
            bool: True if the word contains digits, False otherwise.
        """
        return any(char.isdigit() for char in word)

    def _clean_word(self, word: str) -> str:
        """
        Cleans a word by removing special characters and unnecessary symbols.

        Args:
            word (str): The word to clean.

        Returns:
            str: The cleaned word.
        """
        word = word.replace("$", "").split("=")[-1]
        word = word.replace("'", "")
        while len(word) > 0 and word[-1] != "}" and not word[-1].isdigit():
            word = word[:-1]
        if "{" not in word:
            word = word.replace("}", "")
        word = word.replace("[\\", "")
        return word

    def _get_math_final_result(self, text: str) -> str:
        """
        Extracts the final result from mathematical expressions in the text.

        Args:
            text (str): The input text containing a mathematical expression.

        Returns:
            str: The final result extracted from the text.
        """
        text = text.replace("\f", "\\f")
        text = text.replace("\b", "\\b")
        words = text.split(" ")[::-1]
        for i, _ in enumerate(words):
            words[i] = self._clean_word(words[i])
        text = " ".join(words[::-1])
        return text

    def _remove_boxed(self, text: str) -> str:
        """
        Removes boxed notation from the text.

        Args:
            text (str): The input text containing boxed notation.

        Returns:
            str: The text with boxed notation removed.
        """
        if "oxed" in text:
            text = text.replace(r'"\boxed{', "")
            text = text.replace(r"\boxed{", "")
            text = text.replace(r"\\boxed{", "")
            text = text.replace("\\boxed{", "")
            text = text.replace("\boxed{", "")
            text = text.replace(r"\boxed{", "")
            if text and text[-1] == "}":
                text = text[:-1]
            text = self._clean_word(text)

        return text

    def evaluate(self, data: Dict, args) -> (Dict, Dict):
        """
        Evaluates the predictions against references and calculates metrics.

        Args:
            data (Dict): A dictionary containing 'predictions' and 'references'.
            args: Additional arguments required for evaluation.

        Returns:
            Tuple[Dict, Dict]: A tuple where the first element is the updated data
            dictionary with added scores, and the second element is a dictionary
            containing the F1 score, exact match score, and equality score.
        """
        result = {}
        raw_predictions = data["predictions"]

        predictions = [
            self._get_answer(raw_prediction, args)
            for raw_prediction in raw_predictions
        ]
        references = data["references"]
        references = [
            self._get_answer(reference, args)
            for reference in references
        ]

        f1_scores = [
            f1_score(reference, prediction) for reference,prediction in zip(references, predictions)
        ]
        ems=[exact_match(reference,prediction)for
         reference,prediction in zip(references,predictions)]

        if args.task == "math":
            predictions = [
                self._get_math_final_result(prediction)
                for prediction in predictions
            ]
            references = [
                self._get_math_final_result(reference)
                for reference in references
            ]

            references = [
                self._remove_boxed(reference) for reference in references
            ]

            predictions = [self._remove_boxed(pred) for pred in predictions]
            data["processed_predictions"] = predictions
            data["processed_references"] = references

        equals = [
            is_equiv(prediction, reference)
            for prediction, reference in zip(predictions, references)
        ]
        data["equals"] = equals
        if "fewshot" in data:
            del data["fewshot"]

        result = {
            "f1_score": np.array(f1_scores).mean(),
            "exact_match": np.array(ems).mean(),
            "equality": np.array(equals).mean(),
        }
        return data, result
