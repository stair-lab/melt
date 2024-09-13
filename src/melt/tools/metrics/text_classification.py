"""Module for evaluating text classification models."""

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    f1_score as f1_score_sklearn,
    accuracy_score,
    roc_auc_score,
)
from .utils import normalize_text
from .post_process import softmax_options_prob
from .base import BaseMetric

class TextClassificationMetric(BaseMetric):
    """Evaluate text classification models."""

    def __init__(self, data, args):
        super().__init__(data, args)
        # Ensure 'evaluate' is correctly installed and used, or remove if not needed
        self.roc_auc_score = None  # Remove if not used
        self.data =data

    def evaluate(self, data: Dict, args) -> Tuple[Dict, Dict]:
        """Evaluates the classification performance
        given the predictions, references, and additional arguments.

        Args:
            data (Dict): A dictionary expected to contain keys
            like predictions, references, and option_probs.

            args: Additional arguments including class_names.

        Returns:
            Tuple[Dict, Dict]: The original data dictionary and
            the result dictionary with all the computed metrics.
        """
        result = {}
        raw_predictions = data["predictions"]
        args.class_names = [normalize_text(str(name)) for name in args.class_names]
        predictions = [
            str(self._get_answer(raw_prediction, args))
            for raw_prediction in raw_predictions
        ]
        references = self._normalize_references(data["references"], args)

        result["accuracy"] = accuracy_score(references, predictions)
        result["f1_score"] = f1_score_sklearn(
            references, predictions, average="macro"
        )

        sum_option_probs = [
            [np.array(x).sum() for x in probs]
            for probs in data["option_probs"]
        ]

        probs = softmax_options_prob(sum_option_probs)
        if len(args.class_names) == 2:
            probs = probs[:, 1].reshape(-1, 1)
        labels = np.array([
            args.class_names.index(ref) for ref in references
        ])

        try:
            result["roc_auc"] = roc_auc_score(
                labels, probs, multi_class="ovr", average="macro"
            )
        except (ValueError, TypeError, IndexError) as e:
            print(f"Error calculating ROC AUC: {e}")
            result["roc_auc"] = None

        return data, result
    def reset_data(self, new_data):
        """Resets the data with new data."""
        self.data = new_data
    def _normalize_references(self, references, args):
        """Helper function to normalize references."""

        normalized_references = []
        for reference in references:
            if isinstance(reference, list):
                reference = [normalize_text(str(ref)) for ref in reference]
                first_ref = str(normalize_text(reference[0]))
                answer = self._get_answer(reference, args)
                if answer in reference:
                    normalized_references.append(first_ref)
                else:
                    normalized_references.append(str(reference[0]))
            else:
                normalized_references.append(normalize_text(str(reference)))
        return list(normalized_references)
