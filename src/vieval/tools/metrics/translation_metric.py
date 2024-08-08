import evaluate
from .base import BaseMetric
from hlepor import hlepor_score
from .utils import normalize_text
from typing import Dict


class TranslationMetric(BaseMetric):
    """Evaluate the quality of text translations using metrics like BLEU
    and hLepor."""

    def __init__(self, data, args) -> None:
        self.bleu_metrics = evaluate.load("bleu")
        super().__init__(data, args)

    def evaluate(self, data: Dict, args):
        """Computes the translation quality metrics for
        a set of predictions and references provided in the dictionary.

        Args:
            data (Dict): A dictionary expected to contain two keys:

                - predictions: A list of translated texts generated
                by the translation model.

                - references: A list of reference translations for evaluating
                the quality of the model's predictions.

        Returns:
            1. The original data dictionary, which contains the raw predictions
            and references.

            2. A result dictionary with the following keys:

                - "bleu": The computed BLEU score for the translations.

                - "hLepor": The computed hLepor score for the translations.
        """
        predictions = data["predictions"]
        references = data["references"]
        predictions = [self._get_answer(pre, args) for pre in predictions]
        references = [normalize_text(ref) for ref in references]

        bleu_score = self.bleu_metrics.compute(
            predictions=predictions, references=references
        )["bleu"]
        result = {
            "bleu": bleu_score,
            "hLepor": hlepor_score(references, predictions),
        }
        return data, result
