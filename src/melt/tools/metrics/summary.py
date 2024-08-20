from typing import Dict
from bert_score import BERTScorer
import torch
import evaluate
from .summac.model_summac import SummaCZS
from .data_stats_metric import DataStatsMetric
from .base import BaseMetric
from .utils import normalize_text
import numpy as np


class SummaryMetric(BaseMetric):
    """Evaluate the quality of text summaries."""

    def __init__(self, data, args):
        super().__init__(data, args)
        import warnings

        warnings.filterwarnings("ignore")

        self.rouge = evaluate.load("rouge")
        self.bert_scorer = BERTScorer(
            model_type=args.metric_config["BERTScoreModel"]["model_type"],
            lang=args.lang,
            rescale_with_baseline="baseline_path"
            in args.metric_config["BERTScoreModel"],
            baseline_path=args.metric_config["BERTScoreModel"].get(
                "baseline_path", None
            ),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.data_stats_metric = DataStatsMetric()
        self.summac = SummaCZS(
            granularity="sentence",
            model_name=args.metric_config["SummaCModel"],
            imager_load_cache=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
            args=args,
        )

    def evaluate(self, data: Dict, args) -> (Dict, Dict):
        """Evaluates the generated summaries against reference summaries and
        computes various metrics to assess \
            the quality of the generated summaries.

        Args:
            data (Dict): A dictionary expected to contain \
                original_documents, predictions, and references as keys.

        Returns:
            Returns a tuple containing the original data dictionary and \
                the result dictionary with all the computed metrics.
        """
        inputs = data["original_documents"]
        raw_predictions = data["predictions"][: len(data["references"])]
        predictions = [self._get_answer(r, args) for r in raw_predictions]
        references = [
            str(normalize_text(reference)) for reference in data["references"]
        ]
        result = {}

        print("BERT score")
        p, r, f = self.bert_scorer.score(
            predictions, [[ref] for ref in references], batch_size=args.bs
        )
        result.update(
            {
                "BERTScore-Precision": p[0].item(),
                "BERTScore-Recall": r[0].item(),
                "BERTScore-F1": f[0].item(),
            }
        )

        print("data_stats")
        stats = self.data_stats_metric.evaluate_batch(predictions, inputs)
        result.update(
            {
                "coverage": stats["coverage"],
                "density": stats["density"],
                "compression": stats["compression"],
            }
        )
        print("SummaC")
        result["SummaC"] = np.array(
            self.summac.score(
                [str(normalize_text(input)) for input in inputs], predictions
            )["scores"]
        ).mean()
        print("rouge")
        result.update(
            self.rouge.compute(
                predictions=predictions,
                references=references,
                rouge_types=["rouge1", "rouge2", "rougeL"],
            )
        )
        return data, result
