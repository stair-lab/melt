"""
metric_pipelines.py
"""
from typing import Dict, List
import numpy as np
from melt.tools.metrics.text_classification import TextClassificationMetric
from melt.tools.metrics.calibration_metric import CalibrationMetric
from melt.tools.metrics.language import LanguageMetric
from melt.tools.metrics.ir import InformationRetrievalMetric
from melt.tools.metrics.translation_metric import TranslationMetric
from melt.tools.metrics.question_answering import QAMetric
from melt.tools.metrics.reasoning import ReasoningMetric
from melt.tools.metrics.summary import SummaryMetric
from melt.tools.metrics.bias import BiasMetric
from melt.tools.metrics.toxicity import ToxicityMetric




class MetricPipeline:
    """
    A class for managing and executing various metrics pipelines.
    """
    def __init__(self,config):
        """
        Initialize the MetricPipeline with a configuration.
        """
        self.config = config
        self.metric_classes = {
            "question-answering": [QAMetric, BiasMetric, ToxicityMetric],
            "summarization": [SummaryMetric, BiasMetric, ToxicityMetric],
            "translation": [TranslationMetric, BiasMetric, ToxicityMetric],
            "knowledge-mtpchoice": [
                TextClassificationMetric,
                CalibrationMetric,
            ],
            "knowledge-openended": [QAMetric, BiasMetric, ToxicityMetric],
            "toxicity-detection": [
                TextClassificationMetric,
                CalibrationMetric,
            ],
            "text-classification": [
                TextClassificationMetric,
                CalibrationMetric,
            ],
            "sentiment-analysis": [
                TextClassificationMetric,
                CalibrationMetric,
            ],
            "language-modeling": [LanguageMetric],
            "reasoning": [ReasoningMetric],
            "math": [ReasoningMetric],
            "information-retrieval": [InformationRetrievalMetric],
            "classification": [TextClassificationMetric],
        }
    def _load_metrics(self, data, task_name, config):
        """
        Loads metrics based on the provided configuration.
        """
        class_lst = self.metric_classes[task_name]
        args = config['args']
        args.key_answer = config['answer_key']
        args.class_names = config['class_names']

        obj_lst = [Cls(data, args) for Cls in class_lst]

        return obj_lst


    def run_mean(
        self,
        data,
        task_name: str,
        answer_key: str,
        class_names: List[str],
        *args,
        **kwargs,
    ) -> Dict:
        """
        Computes the mean of metrics based on the provided data and configuration.
        """
        # Define the configuration dictionary
        config = {
            'answer_key': answer_key,
            'class_names': class_names,
            'args': args
        }

        # Call the refactored function
        metric_lst = self._load_metrics(data, task_name, config)
        result = {}
        for metric in metric_lst:
            _, metric_result = metric.evaluate(data, args, **kwargs)
            result.update(metric_result)

        return result

    class TaskConfig:  # pylint: disable=too-few-public-methods
        """
        This class is responsible for storing configuration details for a task.
        """
        def __init__(self, task_name: str, answer_key: str, class_names: List, args):
            self.task_name = task_name
            self.answer_key = answer_key
            self.class_names = class_names
            self.args = args


    def _get_std(self, result_list: List) -> Dict:
        temp = {}
        final_result = {}
        for result in result_list:
            for k in result.keys():
                if result[k]:
                    temp[k] = (
                        temp[k] + [result[k]] if k in temp else [result[k]]
                    )

        final_result.update(
            {
                f"{k}_std": np.array(v).std()
                for k, v in temp.items()
                if v
            }
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

    class BootstrapArgs:
        """
        A class to hold the parameters for bootstrap operations.
        """

        def __init__(self, n_bootstrap: int, p_bootstrap: float):
            self.n_bootstrap = n_bootstrap
            self.p_bootstrap = p_bootstrap

        def __repr__(self):
            return (f"BootstrapArgs(n_bootstrap={self.n_bootstrap}, "
                    f"p_bootstrap={self.p_bootstrap})")

        def to_dict(self):
            """Convert the instance to a dictionary."""
            return {
                'n_bootstrap': self.n_bootstrap,
                'p_bootstrap': self.p_bootstrap
            }

        def update(self, n_bootstrap: int = None, p_bootstrap: float = None):
            """
            Update the parameters of the BootstrapArgs instance.
            """
            if n_bootstrap is not None:
                self.n_bootstrap = n_bootstrap
            if p_bootstrap is not None:
                self.p_bootstrap = p_bootstrap

    def _run_bootstrap(
            self, data, config: dict, **kwargs
        ) -> List[Dict]:
        """
        Performs bootstrap sampling and computes results based on the provided configuration.
        """
        n_data = len(data["predictions"])
        results_lst = []
        n_times = config['args'].n_bootstrap

        for _ in range(n_times):
            indices = np.random.choice(
                np.arange(n_data),
                size=int(config['args'].p_bootstrap * n_data),
                replace=True,
            )
            print(n_data, len(indices))
            sub_data = self._get_subdata(data, n_data, indices)
            result = self.run_mean(
                sub_data,
                config['task_name'],
                config['answer_key'],
                config['class_names'],
                config['args'],
                **kwargs
            )
            results_lst.append(result)
        return results_lst
    def configure(self, new_config):
        """
        Update the configuration of the pipeline.
        """
        self.config = new_config
    def summary(self):
        """
        Provide a summary of the current pipeline configuration.
        """
        return f"Configuration: {self.config}"
    
