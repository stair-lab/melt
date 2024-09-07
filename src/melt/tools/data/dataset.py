"""
This module provides the DatasetWrapper class for loading and managing datasets,
as well as generating prompts based on a configured strategy.
"""

import os
import json
import ast
from .loader import load_a_dataset
from .parser import get_dataset_list


def eval_keys(keys):
    """
    Evaluates the provided keys in the dictionary.

    Args:
        keys (str or list): A key or list of keys to evaluate in the dictionary.

    Returns:
        function: A function to evaluate the keys in the dictionary.
    """
    def eval_x(x):
        if isinstance(keys, str):
            x[keys] = ast.literal_eval(x[keys])
        elif isinstance(keys, list):
            for key in keys:
                x[key] = ast.literal_eval(x[key])
        return x

    return eval_x


class DatasetWrapper:
    """
    A wrapper class for loading datasets, configuring them, and generating prompts
    based on the prompting strategy.
    """
    def __init__(self, args) -> None:
        """
        Initializes the DatasetWrapper with the provided arguments.

        Args:
            args (Namespace): The arguments containing dataset name and configuration.
        """
        self.args = args
        self.datasets = {
            'name': args.dataset_name,
            'training': None,
            'testing': None
        }
        self.dataset_info = None
        self.get_dataset_config()
        self.prompting_strategy = self.dataset_info.prompting_strategy
        self.get_prompt()

    def get_prompt(self):
        """
        Loads the prompt template and calibration instructions based on the dataset
        and prompting strategy.

        Raises:
            ValueError: If the prompting strategy is not supported.
        """
        with open(
            os.path.join(
                self.args.config_dir, self.args.lang, "prompt_template.json"
            ),
            "r", encoding="utf-8"
        ) as f:
            prompt_config = json.load(f)

        prompt_template = prompt_config["PROMPT_TEMPLATE"]
        calibration_instruction = prompt_config["CALIBRATION_INSTRUCTION"]

        if self.prompting_strategy not in [0, 1, 2, 3]:
            raise ValueError("Prompting strategy is not supported")
        task = self.dataset_info.task
        self.prompt = prompt_template[task][self.prompting_strategy]
        self.calibration_prompt = (
            calibration_instruction[task][self.prompting_strategy]
            if task in calibration_instruction else None
        )

    def get_dataset_config(self):
        """
        Loads the dataset configuration and sets up the training and testing datasets.
        """
        self.dataset_info = get_dataset_list(
            dataset_names=[self.datasets['name']],
            dataset_dir=os.path.join(self.args.config_dir, self.args.lang),
        )[0]
        self.datasets['training'], self.datasets['testing'] = load_a_dataset(
            self.dataset_info, self.args
        )

    def get_dataset_testing(self):
        """
        Returns the testing dataset if available.

        Raises:
            ValueError: If the testing dataset is not available.

        Returns:
            Any: The testing dataset.
        """
        if self.datasets['testing'] is None:
            raise ValueError("Dataset testing is not available")
        return self.datasets['testing']

    def get_dataset_training(self):
        """
        Returns the training dataset if available.

        Raises:
            ValueError: If the training dataset is not available.

        Returns:
            Any: The training dataset.
        """
        if self.datasets['training'] is None:
            raise ValueError("Dataset training is not available")
        return self.datasets['training']
