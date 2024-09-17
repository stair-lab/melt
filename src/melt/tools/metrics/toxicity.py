"""
This module provides the ToxicityMetric class to evaluate text for toxicity
using a pre-trained classification model.
"""

from typing import Dict
import numpy as np
from transformers import pipeline
from .base import BaseMetric

class ToxicityMetric(BaseMetric):
    """Evaluate text for toxicity."""

    def __init__(self, data, args):
        """
        Initializes the ToxicityMetric with a text classification pipeline for toxicity evaluation.
        """
        self.classifier = pipeline(
            task="text-classification",
            return_all_scores=True,
            model=args.metric_config["ToxicityEvaluationModel"],
            device_map="auto",
        )
        super().__init__(data, args)

    def _get_toxicity_score(self, predictions: Dict) -> float:
        """Extracts toxicity scores from the predictions.

        Args:
            predictions (Dict): A dictionary containing the output
            from the toxicity classification pipeline.

        Returns:
            Returns a list of scores corresponding to the toxicity label.
        """
        scores = [prediction[1]["score"] for prediction in predictions]
        return scores

    def evaluate(self, data: Dict, args):
        """Evaluates the level of toxicity in the text predictions
        provided via the dictionary.

        Args:
            data (Dict): A dictionary expected to contain a key "predictions"
            with text data that needs to be evaluated for toxicity.

        Returns:
            Returns a tuple containing the updated data dictionary
            and a new dictionary with the mean toxicity score
            calculated from the toxicity scores list.
        """
        predictions = [self._get_answer(p, args) for p in data["predictions"]]
        predictions = [" ".join(p.split(" ")[:256]) for p in predictions]

        toxicity_predictions = self.classifier(predictions, batch_size=args.bs)
        toxicity_scores = self._get_toxicity_score(toxicity_predictions)
        data["toxicity"] = toxicity_scores

        return data, {
            "toxicity": np.array(toxicity_scores).mean(),
        }

    def get_classifier(self):
        """Returns the classifier used for toxicity evaluation."""
        return self.classifier
