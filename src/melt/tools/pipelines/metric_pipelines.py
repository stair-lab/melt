"""
metric_pipelines.py

This module provides functions and classes for performing various tasks
related to data processing, including data manipulation, analysis, and
visualization.

Functions:
- process_data: Processes raw data into a usable format.
- analyze_data: Analyzes processed data and generates insights.

Classes:
- DataProcessor: A class for handling data processing tasks.
- DataAnalyzer: A class for performing data analysis and generating reports.
"""
# Standard library imports
from typing import Dict, List
import numpy as np

# Local application/library imports
from metrics.calibration_metric import CalibrationMetric
from metrics.language import LanguageMetric
from metrics.ir import InformationRetrievalMetric
from metrics.translation_metric import TranslationMetric
from metrics.question_answering import QAMetric
from metrics.reasoning import ReasoningMetric
from metrics.summary import SummaryMetric
from metrics.bias import BiasMetric
from metrics.toxicity import ToxicityMetric
from metrics.text_classification import TextClassificationMetric


class MetricPipeline:
    """
    A class for managing and executing various metrics pipelines.

    This class provides functionality to set up and run a series of metrics 
    pipelines for tasks such as data processing, evaluation, or analysis. 
    It includes methods to initialize the pipeline, add metrics, and execute 
    the pipeline on given data.

    Methods
    -------
    __init__():
        Initializes the MetricPipeline instance.
    add_metric(metric):
        Adds a metric to the pipeline.
    run(data):
        Executes the pipeline on the provided data.
    """
    def __init__(self,config):
        """
        Initialize the MetricPipeline with a configuration.

        Parameters:
        config (dict): A dictionary containing the initial configuration values.
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
            "classification": [TextClassificationMetric],  # Example metric class
            # Add other task names and their corresponding metric classes
        }
    def _load_metrics(self, data, task_name, config):
        """
        Loads metrics based on the provided configuration.

        Parameters
        ----------
        data : Any
            The data to be passed to the metric classes.
        task_name : str
            The name of the task for which metrics are loaded.
        config : dict
            Configuration dictionary with keys:
            - 'answer_key': The key for answers in the data.
            - 'class_names': List of class names relevant to the task.
            - 'args': Additional arguments for metric classes.

        Returns
        -------
        list
            A list of instantiated metric objects.
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

        Parameters
        ----------
        sub_data : Any
            The subset of data to be processed.
        task_name : str
            The name of the task for which metrics are computed.
        answer_key : str
            The key used to extract answers from the data.
        class_names : List[str]
            List of class names relevant to the task.
        args : BootstrapArgs
            Additional arguments required for computing metrics.
        **kwargs : dict
            Any additional keyword arguments to be passed to metric computation methods.

        Returns
        -------
        dict
            A dictionary containing the computed mean metrics.
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

        Attributes:
            task_name (str): The name of the task.
            answer_key (str): The key used for evaluating results.
            class_names (List): A list of class names relevant to the task.
            args: Additional arguments for the task configuration.
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
                for k, v in temp.items()  # Use .items() to iterate over both keys and values
                if v  # Check the value of v instead of temp[k]
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

        Attributes:
            n_bootstrap (int): The number of bootstrap iterations.
            p_bootstrap (float): The proportion of the dataset to use for each bootstrap sample.
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

            Args:
                n_bootstrap (int, optional): New value for the number of bootstrap iterations.
                p_bootstrap (float, optional): New value for the proportion of the dataset to use.
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

        Parameters
        ----------
        data : dict
            The input data containing predictions.
        config : dict
            Configuration dictionary with keys:
            - 'task_name': The name of the task.
            - 'answer_key': The key for answers in the data.
            - 'class_names': List of class names.
            - 'args': Additional arguments for the bootstrap process.
        **kwargs : additional keyword arguments
            Additional parameters for the run_mean method.

        Returns
        -------
        List[Dict]
            A list of results from the bootstrap process.
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

        Parameters:
        new_config (dict): A dictionary containing the new configuration values.

        Returns:
        None
        """
        self.config = new_config
    def summary(self):
        """
        Provide a summary of the current pipeline configuration.

        Returns:
        str: A string representation of the pipeline's current configuration.
        """
        return f"Configuration: {self.config}"
