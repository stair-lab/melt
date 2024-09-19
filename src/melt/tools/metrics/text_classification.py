"test_classification"
from typing import Dict
import numpy as np
import evaluate
from sklearn.metrics import (
    f1_score as f1_score_sklearn,
    accuracy_score,
    roc_auc_score,
)
from melt.tools.metrics.utils import normalize_text
from melt.tools.metrics.post_process import softmax_options_prob
from melt.tools.metrics.base import BaseMetric
class TextClassificationMetric(BaseMetric):
    """Evaluate text classification models."""
    def __init__(self, data, args):
        super().__init__(data, args)
        self.roc_auc_score = evaluate.load("roc_auc", "multiclass")
    def evaluate(self, data: Dict, args) -> tuple[Dict, Dict]:
        """Evaluates the classification performance
        given the predictions, references, and additional arguments.
        Args:
            data (Dict): A dictionary expected to contain keys
            like predictions, references, and option_probs.
        Returns:
            Returns a tuple containing the original data dictionary and
            the result dictionary with all the computed metrics.
        """
        result = {}
        args.class_names = [normalize_text(str(name)) for name in args.class_names]
        predictions = [str(self._get_answer(raw_prediction, args))
                       for raw_prediction in data["predictions"]]
        references = self._process_references(data["references"], predictions)
        result["accuracy"] = accuracy_score(references, predictions)
        result["f1_score"] = f1_score_sklearn(references, predictions, average="macro")
        sum_option_probs = [[np.array(x).sum() for x in option_prob]
                            for option_prob in data["option_probs"]]
        probs = softmax_options_prob(sum_option_probs)
        if len(args.class_names) == 2:
            probs = probs[:, 1].reshape(-1, 1)
        labels = np.array([args.class_names.index(ref) for ref in references])
        try:
            result["roc_auc"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except ValueError as e:
            print(f"ROC AUC calculation failed: {e}")
            result["roc_auc"] = None

        return data, result
    def _process_references(self, references, predictions):
        processed_references = []
        for reference, prediction in zip(references, predictions):
            if isinstance(reference, list):
                reference = [normalize_text(str(ref)) for ref in reference]
                processed_references.append(str(normalize_text(prediction)
                                                 if prediction in reference else reference[0]))
            else:
                processed_references.append(normalize_text(str(reference)))
        return processed_references
