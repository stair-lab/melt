from typing import Dict
from bert_score import BERTScorer
import evaluate
from .summac.model_summac import SummaCZS
from summ_eval.data_stats_metric import DataStatsMetric
from .base import BaseMetric
from .utils import normalize_text


class SummaryMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        import warnings

        warnings.filterwarnings("ignore")

        self.rouge = evaluate.load("rouge")
        self.bert_scorer = BERTScorer(
            model_type="bert-base-multilingual-cased",
            lang="en",
            rescale_with_baseline=True,
            device="cuda",
        )
        self.data_stats_metric = DataStatsMetric()
        self.summac = SummaCZS(
            granularity="sentence",
            model_name="vitc",
            imager_load_cache=False,
            device="cuda",
        )

    def evaluate(self, data: Dict, args) -> (Dict, Dict):
        inputs = data["original_documents"]
        raw_predictions = data["predictions"]
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

        print("rouge")
        result.update(
            self.rouge.compute(
                predictions=predictions,
                references=references,
                rouge_types=["rouge1", "rouge2", "rougeL"],
            )
        )
        return data, result
