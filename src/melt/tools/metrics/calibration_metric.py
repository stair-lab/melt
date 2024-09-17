"""Module for evaluating the calibration of probabilistic models."""


from typing import Dict, List
import numpy as np
try:
    from melt.calibration import get_ece_em, get_ece, get_selective_stats, get_platt_scaler
    print("Import successful")
except ImportError as e:
    print(f"Import error: {e}")
from .utils import normalize_text
from .base import BaseMetric
from .post_process import softmax_options_prob


class CalibrationMetric(BaseMetric):
    """Evaluate the calibration of probabilistic models."""


    def get_cal_score(self, max_probs: List[float], correct: List[int]) -> Dict[str, float]:
        """Calculates various calibration scores based on
        the predicted probabilities (max_probs) and
        the ground truth labels (correct).


        Args:
            max_probs (List[float]): A list of the maximum probabilities
            predicted by the model for each instance.


            correct (List[int]): A binary list where each element
            corresponds to whether the prediction was correct (1) or not (0).


        Returns:
            Dict[str, float]: A dictionary containing ECE scores for 10 bins and 1 bin,
            coverage accuracy area, accuracy in the top 10 percentile,
            and Platt ECE scores for 10 bins and 1 bin.
        """
        max_probs_array = np.array(max_probs)
        correct_array = np.array(correct)


        ece_10_bin = get_ece_em(max_probs_array, correct_array, num_bins=10)
        ece_1_bin = get_ece(max_probs_array, correct_array, num_bins=1)
        coverage_acc_area, acc_top_10_percentile = get_selective_stats(
            max_probs_array, correct_array
        )
        if np.sum(correct_array) == 0 or np.sum(correct_array) == len(correct_array):
            platt_ece_10_bin = 0.0
            platt_ece_1_bin = 0.0
        else:
            platt_scaler, _ = get_platt_scaler(max_probs_array, correct_array, get_clf=False)
            cal_max_probs = platt_scaler(max_probs_array)
            platt_ece_10_bin = get_ece_em(cal_max_probs, correct_array, num_bins=10)
            platt_ece_1_bin = get_ece(cal_max_probs, correct_array, num_bins=1)


        return {
            "ece_10_bin": ece_10_bin,
            "ece_1_bin": ece_1_bin,
            "coverage_acc_area": coverage_acc_area,
            "acc_top_10_percentile": acc_top_10_percentile,
            "platt_ece_10_bin": platt_ece_10_bin,
            "platt_ece_1_bin": platt_ece_1_bin,
        }


    def evaluate(self, data: Dict, args) -> (Dict, Dict):
        """Evaluates the given predictions against the references
        in the dictionary.


        Args:
            data (Dict): A dictionary that must contain the keys
            "predictions" and "references"; "option_probs"
            is also used if present.


        Returns:
            Tuple[Dict, Dict]: Returns a tuple of two dictionaries:
            - The first dictionary is the updated data with
            additional key "max_probs".
            - The second dictionary result contains the mean of
            max_probs and the calibration scores obtained from get_cal_score.
        """
        result = {}
        raw_predictions = data["predictions"]
        predictions = [
            self._get_answer(raw_prediction, args)
            for raw_prediction in raw_predictions
        ]
        references = data["references"]


        accuracy = [
            int(normalize_text(str(pred)) == normalize_text(str(ref)))
            for pred, ref in zip(predictions, references)
        ]
        option_probs = data.get("option_probs", [])
        if option_probs:
            sum_option_probs = [
                [np.array(x).sum() for x in option_probs[i]]
                for i in range(len(option_probs))
            ]
        else:
            sum_option_probs = []


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
