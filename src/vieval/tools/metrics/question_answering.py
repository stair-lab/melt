from typing import Dict
import numpy as np
from .basic_metrics import exact_match, f1_score
from .base import BaseMetric
from .utils import normalize_text


class QAMetric(BaseMetric):
    """Evaluate the performance of a question-answering (QA) system."""

    # def __init__(self):
    #     super().__init__()

    def evaluate(self, data: Dict, args) -> (Dict, Dict):
        """Returns evaluation results for QA predictions.

        Args:
            data (Dict): A dictionary expected to contain the keys "predictions" and "references". It represents the dataset being evaluated, with "predictions" containing the model's answers to the questions, and "references" containing the ground truth answers.
        """
        result = {}
        raw_predictions = data["predictions"]
        predictions = [
            self._get_answer(raw_prediction, args) for raw_prediction in raw_predictions
        ]
        references = data["references"]

        f1_scores = [f1_score(*batch) for batch in zip(references, predictions)]
        em_scores = [
            exact_match(normalize_text(pred), normalize_text(ref))
            for ref, pred in zip(references, predictions)
        ]

        data["f1_score"] = f1_scores
        data["exact_match"] = em_scores
        result = {
            "f1_score": np.array(f1_scores).mean(),
            "exact_match": np.array(em_scores).mean(),
        }
        return data, result
