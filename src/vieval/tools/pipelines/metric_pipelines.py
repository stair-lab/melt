import argparse
import os
from ..metrics.text_classification import TextClassificationMetric
from ..metrics.calibration_metric import CalibrationMetric
from ..metrics.language import LanguageMetric
from ..metrics.ir import InformationRetrievalMetric
from ..metrics.translation_metric import TranslationMetric
from ..metrics.question_answering import QAMetric
from ..metrics.reasoning import ReasoningMetric
from ..metrics.summary import SummaryMetric
from ..metrics.bias import BiasMetric
from ..metrics.toxicity import ToxicityMetric
from time import time
import logging

from typing import Dict, List
import numpy as np


class MetricPipeline:
    def __init__(self):

        self.metric_classes = {
            "question-answering": [QAMetric, BiasMetric, ToxicityMetric],
            "summarization": [SummaryMetric, BiasMetric, ToxicityMetric],
            "translation": [TranslationMetric, BiasMetric, ToxicityMetric],
            "knowledge-mtpchoice": [QAMetric, BiasMetric, ToxicityMetric],
            "knowledge-openended": [QAMetric, BiasMetric, ToxicityMetric],
            "toxicity-detection": [TextClassificationMetric, CalibrationMetric],
            "text-classification": [TextClassificationMetric, CalibrationMetric],
            "sentiment-analysis": [TextClassificationMetric, CalibrationMetric],
            "language-modeling": [LanguageMetric],
            "reasoning": [ReasoningMetric],
            "information-retrieval": [InformationRetrievalMetric],
        }

    def _load_metrics(self, data, task_name, answer_key, class_names, args):

        class_lst = self.metric_classes[task_name]
        args.key_answer = answer_key
        args.class_names = class_names

        obj_lst = [Cls(data, args) for Cls in class_lst]

        return obj_lst

    def run_mean(
        self, data, task_name: str, answer_key: str, class_names: List, args
    ) -> Dict:
        metric_lst = self._load_metrics(data, task_name, answer_key, class_names, args)
        result = {}
        for metric in metric_lst:
            _, metric_result = metric.evaluate(data, args)
            result.update(metric_result)

        return result

    def run_std(
        self, data, task_name: str, answer_key: str, class_names: List, args
    ) -> Dict:
        result_lst = self._run_bootrap(data, task_name, answer_key, args)
        final_result = self._get_std(result_lst)

        return final_result

    def _get_std(self, result_list: List) -> Dict:
        temp = {}
        final_result = {}
        for result in result_list:
            for k in result.keys():
                if result[k]:
                    temp[k] = temp[k] + [result[k]] if k in temp else [result[k]]

        final_result.update(
            {f"{k}_std": np.array(temp[k]).std() for k in temp.keys() if temp[k]}
        )
        return final_result

    def _get_subdata(self, data: Dict, n: int, indices) -> Dict:
        sub_data = {}
        for key in data.keys():
            if isinstance(data[key], list) and len(data[key]) == n:
                sub_data[key] = [data[key][i] for i in indices]
                print(key, len(sub_data[key]))
            else:
                sub_data[key] = data[key]

        return sub_data

    def _run_bootrap(self, data, task_name, answer_key, class_names, args) -> Dict:
        n_data = len(
            data["predictions"]
        )  # if 'predictions' in data else len(data['prediction'])
        results_lst = []
        n_times = args.n_bootstrap
        for i in range(n_times):
            indices = np.random.choice(
                np.arange(n_data), size=int(args.p_bootstrap * n_data), replace=True
            )
            print(n_data, len(indices))
            sub_data = self._get_subdata(data, n_data, indices)
            result = self.run_mean(sub_data, task_name, answer_key, class_names, args)
            results_lst.append(result)

        return results_lst
