import evaluate
from .base import BaseMetric
from hlepor import hlepor_score
from .utils import normalize_text
from typing import Dict


class TranslationMetric(BaseMetric):
    def __init__(self) -> None:
        self.bleu_metrics = evaluate.load("bleu")

    def evaluate(self, data: Dict, args):
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
