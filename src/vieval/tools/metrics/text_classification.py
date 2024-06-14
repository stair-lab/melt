from typing import Dict
import numpy as np
from .utils import normalize_text
from .post_process import softmax_options_prob
import evaluate
from .base import BaseMetric
from sklearn.metrics import (
    f1_score as f1_score_sklearn,
    accuracy_score,
    roc_auc_score,
)


class TextClassificationMetric(BaseMetric):
    def __init__(self, data, args):
        super().__init__(data, args)
        self.roc_auc_score = evaluate.load("roc_auc", "multiclass")

    def evaluate(self, data: Dict, args, **kwargs) -> None:
        result = {}
        raw_predictions = data["predictions"]
        args.class_names = [normalize_text(str(name)) for name in args.class_names]
        predictions = [
            str(self._get_answer(raw_prediction, args))
            for raw_prediction in raw_predictions
        ]
        references = data["references"]

        for i, (prediction, reference) in enumerate(zip(predictions, references)):
            if isinstance(reference, list):
                reference = [normalize_text(str(ref)) for ref in reference]
                if prediction in reference:
                    references[i] = str(normalize_text(prediction))
                else:
                    references[i] = str(references[i][0])
            else:
                references[i] = normalize_text(str(references[i]))

        references = [str(ref) for ref in references]
        result["accuracy"] = accuracy_score(
            [ref for ref in references], [pred for pred in predictions]
        )
        f1_score = f1_score_sklearn(
            [ref for ref in references], [pred for pred in predictions], average="macro"
        )
        result["f1_score"] = f1_score

        sum_option_probs = []
        for i in range(len(data["option_probs"])):
            sum_option_probs.append(
                [np.array(x).sum() for x in data["option_probs"][i]]
            )

        probs = softmax_options_prob(sum_option_probs)
        if len(args.class_names) == 2:
            probs = probs[:, 1].reshape(-1, 1)
        labels = np.array([args.class_names.index(ref) for ref in references])
        try:
            roc_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            # self.roc_auc_score.compute(
            #     references=labels,
            #     prediction_scores=probs,
            #     multi_class="ovr",
            #     average="macro",
            # )
            result["roc_auc"] = roc_auc
        except Exception as e:
            print(e)
            result["roc_auc"] = None

        return data, result
