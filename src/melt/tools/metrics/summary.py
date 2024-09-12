"""
This module provides utilities for evaluating the quality of text summaries.

It includes the `SummaryMetric` class, which provides functionality to evaluate
text summaries using various metrics, including BERTScore, ROUGE, data statistics,
and SummaC. The `SummaryMetric` class is designed to be initialized with data and
configuration arguments, and it provides methods to evaluate summaries and
generate reports.

Classes:
- SummaryMetric: Evaluates the quality of text summaries using various metrics.

Imports:
- warnings: For controlling warning messages.
- Dict: Type hint for dictionaries.
- BERTScorer: For computing BERT scores for text.
- torch: For handling GPU acceleration.
- evaluate: For loading evaluation metrics such as ROUGE.
- numpy: For numerical operations.
- SummaCZS: A class for SummaC evaluation.
- DataStatsMetric: A class for evaluating data statistics.
- BaseMetric: A base class for metrics.
- normalize_text: Utility function for normalizing text.
"""
import warnings
from typing import Dict

try:
    from bert_score import BERTScorer
except ImportError as e:
    raise ImportError("The 'bert_score' module is required but not installed. ") from e
import torch

try:
    import evaluate
except ImportError as e:
    raise ImportError("The 'evaluate' module is required but not installed.") from e

import numpy as np
from .summac.model_summac import SummaCZS
from .data_stats_metric import DataStatsMetric
from .base import BaseMetric
from .utils import normalize_text

class SummaryMetric(BaseMetric):
    """Evaluate the quality of text summaries."""

    def __init__(self, data, args):
        super().__init__(data, args)

        warnings.filterwarnings("ignore")

        self.rouge = evaluate.load("rouge")
        self.bert_scorer = BERTScorer(
            model_type=args.metric_config["BERTScoreModel"]["model_type"],
            lang=args.lang,
            rescale_with_baseline="baseline_path" in args.metric_config["BERTScoreModel"],
            baseline_path=args.metric_config["BERTScoreModel"].get("baseline_path", None),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.data_stats_metric = DataStatsMetric()
        self.summac = SummaCZS(
            # Adjust parameters according to the SummaCZS documentation
            config=args.metric_config["SummaCModel"],  # Example parameter
            # Remove or adjust other parameters based on the class definition
        )

    def evaluate(self, data: Dict, args) -> (Dict, Dict):
        """Evaluates the generated summaries against reference summaries and computes 
        various metrics to assess the quality of the generated summaries."""
        inputs = data["original_documents"]
        raw_predictions = data["predictions"][: len(data["references"])]
        predictions = [self._get_answer(r, args) for r in raw_predictions]
        references = [str(normalize_text(reference)) for reference in data["references"]]
        result = {}

        print("BERT score")
        p, r, f = self.bert_scorer.score(
            predictions, [[ref] for ref in references], batch_size=args.bs
        )
        result.update({
            "BERTScore-Precision": p[0].item(),
            "BERTScore-Recall": r[0].item(),
            "BERTScore-F1": f[0].item(),
        })

        print("data_stats")
        stats = self.data_stats_metric.evaluate_batch(predictions, inputs)
        result.update({
            "coverage": stats["coverage"],
            "density": stats["density"],
            "compression": stats["compression"],
        })

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

    def calculate_score(self, summary):
        """Calculate the score for the given summary."""
        # Implementation here

    def report(self):
        """Generate a report based on the calculated scores."""
        # Implementation here
