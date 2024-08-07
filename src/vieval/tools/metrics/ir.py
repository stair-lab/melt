from typing import Dict, List
import numpy as np
from .base import BaseMetric
from ranx import Qrels, Run, evaluate as ranx_evaluate


class InformationRetrievalMetric(BaseMetric):
    """Evaluate information retrieval systems."""

    def _get_qrel(self, references: List[Dict]) -> Qrels:
        """Processes a list of reference dictionaries to create a Qrels object, which represents the relevance judgments (i.e., which documents are relevant to which queries).

        Args:
            references (List[Dict]): A list of dictionaries, each containing an "id" key representing the query ID and a "references" key containing a list of document IDs that are relevant to the query.
        """
        relevant_dict = {}
        for reference in references:
            query_id = str(reference["id"])
            if query_id not in relevant_dict:
                relevant_dict[query_id] = {}
            for doc_id in reference["references"]:
                relevant_dict[query_id][str(doc_id)] = 1

        qrels = Qrels(relevant_dict)
        return qrels

    def _get_prob_from_log_prob(
        self,
        score: float,
        is_positive_predict: bool,
    ) -> float:
        """Converts a log probability score into a regular probability.

        Args:
            score (float): The log probability score.

            is_positive_predict (bool): A boolean indicating whether the prediction is positive.

        Returns:
            float: If the prediction is not positive, the probability is adjusted by subtracting it from 1.
        """
        prob = np.exp(score)
        prob = 1 - prob if not is_positive_predict else prob
        return prob

    def _get_run(self, predictions: List[Dict], k: int, args) -> Run:
        """Processes a list of prediction dictionaries to create a Run object, which represents the system's ranked list of documents for each query.

        Args:
            predictions (List[Dict]): A list of dictionaries, each containing a "query_id", "prediction", and "calib_probs".

            k (int): An integer representing the number of top documents to consider for each query.
        """
        run_dict = {}
        for prediction in predictions:
            query_id = str(prediction["query_id"])
            if query_id not in run_dict:
                run_dict[query_id] = {}

            predict = self._get_answer(prediction["prediction"], args)
            is_positive_predict = predict == "yes"
            try:
                log_prob = (
                    prediction["calib_probs"][0][0][0]
                    if is_positive_predict
                    else prediction["calib_probs"][1][0][0]
                )
            except:
                log_prob = 0
            prob = self._get_prob_from_log_prob(log_prob, is_positive_predict)
            if len(run_dict[query_id]) < k:
                run_dict[query_id][str(prediction["passage_id"])] = prob

        run = Run(run_dict)
        return run

    def evaluate(self, data: Dict, args, **kwargs) -> (Dict, Dict):
        """Evaluates the predictions using relevance judgments and computes various metrics.

        Args:
            data (Dict): A dictionary containing predictions to be evaluated.
        """
        result = {}

        refenreces = kwargs["ref_dataset"]
        predictions = data["predictions"]

        qrels = self._get_qrel(refenreces)

        for mode in ["regular", "boosted"]:
            if mode == "regular":
                k = 30
            else:
                k = 9999
            run = self._get_run(predictions, k, args)
            result[f"{mode}_recall@10"] = ranx_evaluate(
                qrels, run, "recall@10", make_comparable=True
            )
            result[f"{mode}_precision@10"] = ranx_evaluate(
                qrels, run, "precision@10", make_comparable=True
            )
            result[f"{mode}_hit_rate@10"] = ranx_evaluate(
                qrels, run, "hit_rate@10", make_comparable=True
            )
            result[f"{mode}_mrr@10"] = ranx_evaluate(
                qrels, run, "mrr@10", make_comparable=True
            )
            result[f"{mode}_ndcg@10"] = ranx_evaluate(
                qrels, run, "ndcg@10", make_comparable=True
            )
            print(result)
        return data, result
