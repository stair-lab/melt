"""Module for evaluating information retrieval systems."""

from typing import Dict, List
import numpy as np
try:
    from ranx import Qrels, Run, evaluate as ranx_evaluate
except ImportError as e:
    raise ImportError(
        "Failed to import 'ranx'. Ensure that 'ranx' is installed in your environment. "
        "You can install it using 'pip install ranx'. Original error: " + str(e)
    ) from e

from .base import BaseMetric  # Local import

class InformationRetrievalMetric(BaseMetric):
    """Evaluate information retrieval systems."""

    def _get_qrel(self, references: List[Dict]) -> Qrels:
        """Processes a list of reference dictionaries to create a Qrels object.

        Args:
            references (List[Dict]): List of dictionaries with "id" and "references" keys.

        Returns:
            Qrels: An object representing relevance judgments.
        """
        relevant_dict = {}
        for reference in references:
            query_id = str(reference["id"])
            relevant_dict.setdefault(query_id, {})
            for doc_id in reference["references"]:
                relevant_dict[query_id][str(doc_id)] = 1

        return Qrels(relevant_dict)

    def _get_prob_from_log_prob(self, score: float, is_positive_predict: bool) -> float:
        """Converts a log probability score into a regular probability.

        Args:
            score (float): The log probability score.
            is_positive_predict (bool): Whether the prediction is positive.

        Returns:
            float: Adjusted probability.
        """
        prob = np.exp(score)
        return prob if is_positive_predict else 1 - prob

    def _get_run(self, predictions: List[Dict], k: int, args) -> Run:
        """Processes predictions to create a Run object.

        Args:
            predictions (List[Dict]): List of dictionaries with "query_id", "prediction", 
            and "calib_probs" keys.
            k (int): Number of top documents to consider.
            args: Additional arguments.

        Returns:
            Run: An object representing the ranked list of documents.
        """
        run_dict = {}
        for prediction in predictions:
            query_id = str(prediction["query_id"])
            run_dict.setdefault(query_id, {})

            predict = self._get_answer(prediction["prediction"], args)
            is_positive_predict = predict == "yes"

            try:
                log_prob = (
                    prediction["calib_probs"][0][0][0]
                    if is_positive_predict
                    else prediction["calib_probs"][1][0][0]
                )
            except (IndexError, KeyError):
                log_prob = 0

            prob = self._get_prob_from_log_prob(log_prob, is_positive_predict)
            if len(run_dict[query_id]) < k:
                run_dict[query_id][str(prediction["passage_id"])] = prob

        return Run(run_dict)

    def evaluate(self, data: Dict, args, **kwargs) -> (Dict, Dict):
        """Evaluates predictions and computes various metrics.

        Args:
            data (Dict): Dictionary with predictions to be evaluated.
            args: Additional arguments.
            **kwargs: Additional keyword arguments including "ref_dataset".

        Returns:
            Tuple[Dict, Dict]: Updated data with metrics results.
        """
        result = {}

        references = kwargs.get("ref_dataset", [])
        if not references:
            raise ValueError("Reference dataset is missing in kwargs")

        predictions = data.get("predictions", [])
        qrels = self._get_qrel(references)

        for mode in ["regular", "boosted"]:
            k = 30 if mode == "regular" else 9999
            run = self._get_run(predictions, k, args)

            for metric in [
                "recall@10", "precision@10", "hit_rate@10", "mrr@10", "ndcg@10"
            ]:
                result[f"{mode}_{metric}"] = ranx_evaluate(
                    qrels, run, metric, make_comparable=True
                )
        print(result)
        return data, result
