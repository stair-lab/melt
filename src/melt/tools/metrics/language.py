"""This module defines metrics for evaluating language generation tasks."""

from typing import Dict, List
import math
import numpy as np

# Attempt to import third-party libraries
try:
    import evaluate
except ImportError as e:
    raise ImportError("The 'evaluate' package is required but could not be imported. "
                      "Please install it using 'pip install evaluate'.") from e

try:
    import Levenshtein
except ImportError as e:
    raise ImportError("The 'Levenshtein' package is required but could not be imported. "
                      "Please install it using 'pip install python-Levenshtein'.") from e

from .base import BaseMetric
from .basic_metrics import exact_match
from .utils import normalize_text


class LanguageMetric(BaseMetric):
    """Evaluate language generation tasks."""

    def __init__(self, data, args) -> None:
        """Initialize the metric with data and arguments."""
        self.cer_metrics = evaluate.load("cer")
        self.wer_metrics = evaluate.load("wer")
        super().__init__(data, args)

    def get_num_bytes(self, tokens: List[str]) -> int:
        """Calculate the total number of bytes of a list of tokens
        when encoded in UTF-8.

        Args:
            tokens (List[str]): A list of string tokens for which the byte
            length is to be calculated.

        Returns:
            int: Total number of bytes.
        """
        return sum(len(bytes(token, encoding="utf-8")) for token in tokens)

    def _compute_perplexity(self, prediction: str, generation_prob: List[float]) -> tuple:
        """Compute perplexity for a given prediction and generation probabilities."""
        logprob = np.array(generation_prob).sum()
        num_perplexity_tokens = len(generation_prob)
        num_bytes = self.get_num_bytes(prediction.split(" "))
        perplexity = math.e ** (-logprob / num_perplexity_tokens)
        bits_per_byte = -logprob / num_bytes / math.log(2)
        logprob_per_byte = logprob / num_bytes
        return perplexity, bits_per_byte, logprob_per_byte

    def evaluate(self, data: Dict, args) -> tuple:
        """Evaluate predictions against references and compute various metrics.

        Args:
            data (Dict): A dictionary that must contain keys
            "predictions", "references", and "generation_probs".

        Returns:
            Tuple[Dict, Dict]: Updated data dictionary with raw metric scores
            and a result dictionary with average scores.
        """
        predictions = [self._get_answer(pred, args) for pred in data["predictions"]]
        references = [normalize_text(ref) for ref in data["references"]]

        em_scores = [
            exact_match(pred, ref)
            for ref, pred in zip(references, predictions)
        ]
        cer_score = self.cer_metrics.compute(
            predictions=predictions, references=references
        )
        wer_score = self.wer_metrics.compute(
            predictions=predictions, references=references
        )

        ced_scores = [
            Levenshtein.distance(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        wed_scores = [
            Levenshtein.distance(
                np.array(pred.split(" ")), np.array(ref.split(" "))
            )
            for pred, ref in zip(predictions, references)
        ]

        perplexity_scores, bits_per_byte, logprob_per_byte = zip(
            *[self._compute_perplexity(pred, gen_prob)
              for pred, gen_prob in zip(data["predictions"], data["generation_probs"])]
        )

        data.update(
            {
                "average_exact_match": em_scores,
                "ced": ced_scores,
                "wed": wed_scores,
                "perplexity": perplexity_scores,
                "bits_per_byte": bits_per_byte,
                "logprob_per_byte": logprob_per_byte,
            }
        )
        result = {
            "average_exact_match": np.mean(em_scores),
            "cer": cer_score,
            "wer": wer_score,
            "ced": np.mean(ced_scores),
            "wed": np.mean(wed_scores),
            "perplexity": np.mean(perplexity_scores),
            "bits_per_byte": np.mean(bits_per_byte),
            "logprob_per_byte": np.mean(logprob_per_byte),
        }

        return data, result
