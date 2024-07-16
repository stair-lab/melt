from typing import Dict
import numpy as np
from .basic_metrics import exact_match, f1_score
from .base import BaseMetric
import string
import random
import Levenshtein

escape_dict = {
    "\a": r"\a",
    "\b": r"\b",
    "\f": r"\f",
    "\n": r"\n",
    "\r": r"\r",
    "\t": r"\t",
    "\v": r"\v",
}


class ReasoningMetric(BaseMetric):
    """Evaluate reasoning capabilities, particularly in scenarios that may involve mathematical reasoning."""

    def __init__(self):
        super().__init__()

    def equal(self, prediction: str, refenrence: str, threshold: int = 0.9) -> float:
        """Evaluates whether the prediction is sufficiently close to the refenrence using a similarity threshold. It employs the Levenshtein ratio (a measure of the similarity between two strings) and returns 1 if the ratio exceeds the threshold, indicating a match, otherwise 0.

        Args:
            prediction (str): The predicted answer.

            refenrence (str): The reference or ground truth answer.

            threshold (int, optional): A similarity threshold for comparing the prediction and reference, defaulting to 0.9.
        """
        if Levenshtein.ratio(refenrence, prediction) > threshold:
            return 1
        else:
            return 0

    def _has_numbers(self, word: str):
        """Checks if a given word contains any digit characters.

        Args:
            word (str): The string to be checked for numeric digits.
        """
        return any(char.isdigit() for char in word)

    def _get_math_final_result(self, text: str) -> str:
        """Extract and return the numerical result from a mathematical expression represented as a string, text. If no numbers are found, it generates a random 4-letter string.

        Args:
            text (str): The string containing the mathematical expression.
        """
        words = text.split(" ")[::-1]

        for word in words:
            word = word.replace("$", "").split("=")[-1]
            while word and word[0] in escape_dict and not word[0].isdigit():
                word = word[1:]
            while word and word[-1] != "}" and not word[-1].isdigit():
                word = word[:-1]
            if self._has_numbers(word):
                return word
        return "".join(random.choice(string.ascii_uppercase) for _ in range(4))

    def _remove_boxed(self, text: str) -> str:
        """Removes LaTeX boxed annotations from a given string.

        Args:
            text (str): The string from which LaTeX boxed annotations are to be removed.
        """
        text = text.replace(r"\boxed{", "")
        text = text.replace("\boxed{", "")
        if text[-1] == "}":
            text = text[:-1]
        return text

    def evaluate(self, data: Dict, args) -> (Dict, Dict):
        """Evaluates predictions against references contained within the dictionary using various metrics.

        Args:
            data (Dict): A dictionary that must contain the keys "predictions" and "references".
        """
        result = {}
        raw_predictions = data["predictions"]
        predictions = [
            self._get_answer(raw_prediction, args) for raw_prediction in raw_predictions
        ]
        references = data["references"]
        references = [
            self._get_answer(
                "{" + f"'{args.key_answer}'" + ":" + f"'{reference}'" + "}", args
            )
            for reference in references
        ]
        if "math" in args.filepath:
            references = [self._remove_boxed(reference) for reference in references]
            predictions = [self._remove_boxed(pred) for pred in predictions]

        f1_scores = [f1_score(*batch) for batch in zip(references, predictions)]
        ems = [exact_match(*batch) for batch in zip(references, predictions)]
        data["f1_score"] = f1_scores
        data["em"] = ems
        if "math" in args.filepath:
            predictions = [
                self._get_math_final_result(prediction) for prediction in predictions
            ]
            references = [
                self._get_math_final_result(reference) for reference in references
            ]
        equals = [
            self.equal(prediction, refenrence)
            for prediction, refenrence in zip(predictions, references)
        ]
        data["equals"] = equals
        result = {
            "f1_score": np.array(f1_scores).mean(),
            "exact_match": np.array(ems).mean(),
            "equality": np.array(equals).mean(),
        }

        return data, result
