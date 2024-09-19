"language"
from typing import Dict, List, Tuple
import math
import numpy as np
import evaluate
import Levenshtein
from melt.tools.metrics.base import BaseMetric
from melt.tools.metrics.basic_metrics import exact_match
from melt.tools.metrics.utils import normalize_text

class LanguageMetric(BaseMetric):
    """Evaluate language generation tasks."""

    def __init__(self, data, args) -> None:
        self.cer_metrics = evaluate.load("cer")
        self.wer_metrics = evaluate.load("wer")
        super().__init__(data, args)

    def get_num_bytes(self, tokens: List[str]) -> int:
        """Calculates the total number of bytes of a list of tokens
        when encoded in UTF-8.

        Args:
            tokens (List[str]): A list of string tokens for which the byte
            length is to be calculated.
        """
        return sum(len(bytes(token, encoding="utf-8")) for token in tokens)

    def compute_edit_distances(self, predictions: List[str],
                                references: List[str]) -> Tuple[List[int], List[int]]:
        """Compute Character Edit Distance (CED) and Word Edit Distance (WED)"""
        ced_scores = [Levenshtein.distance(pred, ref) for pred, ref in zip(predictions, references)]
        wed_scores = [Levenshtein.distance(pred.split(), ref.split())
                      for pred, ref in zip(predictions, references)]
        return ced_scores, wed_scores

    def compute_perplexity_metrics(
            self, predictions: List[str],
            generation_probs: List[List[float]]) ->Tuple[List[float], List[float], List[float]]:
        """Compute perplexity, bits per byte, and log probability per byte"""
        perplexity_scores, bits_per_byte, logprob_per_byte = [], [], []
        for prediction, generation_prob in zip(predictions, generation_probs):
            logprob = np.array(generation_prob).sum()
            num_perplexity_tokens = len(generation_prob)
            num_bytes = self.get_num_bytes(prediction.split())

            perplexity_scores.append(math.e ** (-logprob / num_perplexity_tokens))
            bits_per_byte.append(-logprob / num_bytes / math.log(2))
            logprob_per_byte.append(logprob / num_bytes)

        return perplexity_scores, bits_per_byte, logprob_per_byte

    def evaluate(self, data: Dict, args) -> Tuple[Dict, Dict]:
        """Evaluates the predictions against references and
        computes various metrics."""
        predictions = [self._get_answer(pred, args) for pred in data["predictions"]]
        references = [normalize_text(ref) for ref in data["references"]]

        em_scores = [exact_match(pred, ref) for ref, pred in zip(references, predictions)]
        cer_score = self.cer_metrics.compute(predictions=predictions, references=references)
        wer_score = self.wer_metrics.compute(predictions=predictions, references=references)

        ced_scores, wed_scores = self.compute_edit_distances(predictions, references)
        perplexity_scores, bits_per_byte, logprob_per_byte = (
            self.compute_perplexity_metrics(data["predictions"], data["generation_probs"]))
        data.update({
            "average_exact_match": em_scores,
            "ced": ced_scores,
            "wed": wed_scores,
            "perplexity": perplexity_scores,
            "bits_per_byte": bits_per_byte,
            "logprob_per_byte": logprob_per_byte,
        })
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
