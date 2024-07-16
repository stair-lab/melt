from typing import Dict
import calibration as cal
import numpy as np
from .utils import normalize_text
from .base import BaseMetric
from .post_process import softmax_options_prob
from typing import List


class CalibrationMetric(BaseMetric):
    """Evaluate the calibration of probabilistic models"""

    # def __init__(self) -> None:
    #     pass

    def get_cal_score(self, max_probs: List[float], correct: List[int]):
        """Calculates various calibration scores based on the predicted probabilities (max_probs) and the ground truth labels (correct).

        Args:
            max_probs (List[float]): A list of the maximum probabilities predicted by the model for each instance.

            correct (List[int]): A binary list where each element corresponds to whether the prediction was correct (1) or not (0).

        Returns:
            A dictionary containing ECE scores for 10 bins and 1 bin, coverage accuracy area, accuracy in the top 10 percentile, and Platt ECE scores for 10 bins and 1 bin.
        """
        ece_10_bin = cal.get_ece_em(max_probs, correct, num_bins=10)
        ece_1_bin = cal.get_ece(max_probs, correct, num_bins=1)
        coverage_acc_area, acc_top_10_percentile = cal.get_selective_stats(
            max_probs, correct
        )
        if np.sum(correct) == 0 or np.sum(correct) == len(correct):
            platt_ece_10_bin = 0.0
            platt_ece_1_bin = 0.0
        else:
            platt_scaler, clf = cal.get_platt_scaler(
                np.array(max_probs), np.array(correct), get_clf=True
            )
            cal_max_probs = platt_scaler(np.array(max_probs))
            platt_ece_10_bin = cal.get_ece_em(cal_max_probs, correct, num_bins=10)
            platt_ece_1_bin = cal.get_ece(cal_max_probs, correct, num_bins=1)

        return {
            "ece_10_bin": ece_10_bin,
            "ece_1_bin": ece_1_bin,
            "coverage_acc_area": coverage_acc_area,
            "acc_top_10_percentile": acc_top_10_percentile,
            "platt_ece_10_bin": platt_ece_10_bin,
            "platt_ece_1_bin": platt_ece_1_bin,
        }

    def evaluate(self, data: Dict, args, **kwargs) -> (Dict, Dict):
        """Evaluates the given predictions against the references in the dictionary.

        Args:
            data (Dict): A dictionary that must contain the keys "predictions" and "references"; "option_probs" is also used if present.

        Returns:
            Returns a tuple of two dictionaries:
            - The first dictionary is the updated data with additional key "max_probs".
            - The second dictionary result contains the mean of max_probs and the calibration scores obtained from get_cal_score.
        """
        result = {}
        raw_predictions = data["predictions"]
        predictions = [
            self._get_answer(raw_prediction, args) for raw_prediction in raw_predictions
        ]
        references = data["references"]

        accuracy = [
            int(normalize_text(str(pred)) == normalize_text(str(ref)))
            for pred, ref in zip(predictions, references)
        ]
        sum_option_probs = []
        for i in range(len(data["option_probs"])):
            sum_option_probs.append(
                [np.array(x).sum() for x in data["option_probs"][i]]
            )

        if "gpt" in args.filepath:
            probs = softmax_options_prob(sum_option_probs)
            probs = np.zeros_like(probs)
            labels = np.array([args.class_names.index(str(ref)) for ref in references])

            for i, label in enumerate(labels):
                probs[i][label] = 1
        else:
            probs = softmax_options_prob(sum_option_probs)

        max_probs = np.max(probs, axis=1)
        data["max_probs"] = list(max_probs)
        result["max_probs"] = max_probs.mean()
        result.update(self.get_cal_score(max_probs, accuracy))

        return data, result
