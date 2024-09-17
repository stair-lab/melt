"WRAPPER"
import os
import sys
import json
import ast
from typing import Dict, Any, Optional, Tuple
from argparse import Namespace
from melt.tools.data.parser import get_dataset_list

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def load_a_dataset() -> Tuple[Any, Any]:
    """
    Placeholder function for loading a dataset.
    Returns a tuple of (training_dataset, testing_dataset).
    """
    # Implement dataset loading logic here
    training_dataset = None  # Replace with actual loading logic
    testing_dataset = None   # Replace with actual loading logic
    return training_dataset, testing_dataset

def eval_keys(keys: str | list[str]) -> callable:
    """
    Returns a function that evaluates the provided keys in the dictionary.
    """
    def eval_x(x: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(keys, str):
            x[keys] = ast.literal_eval(x[keys])
        elif isinstance(keys, list):
            for key in keys:
                x[key] = ast.literal_eval(x[key])
        return x
    return eval_x

class DatasetWrapper:
    """
    A wrapper class for managing datasets and generating prompts.
    """
    def __init__(self, args: Namespace) -> None:
        """
        Initializes the DatasetWrapper with the provided arguments.
        """
        self.args = args
        self.datasets: Dict[str, Optional[Any]] = {
            'name': args.dataset_name,
            'training': None,
            'testing': None
        }
        self.dataset_info: Optional[Dict[str, Any]] = None
        self.get_dataset_config()
        self.prompting_strategy: int = self.dataset_info['prompting_strategy']
        self.get_prompt()
    def get_prompt(self) -> None:
        """
        Get the prompt template and calibration instruction based on the prompting strategy.
        """
        prompt_config_path = os.path.join(
            self.args.config_dir, self.args.lang, "prompt_template.json"
        )
        with open(prompt_config_path, "r", encoding="utf-8") as f:
            prompt_config = json.load(f)
        prompt_template = prompt_config["PROMPT_TEMPLATE"]
        calibration_instruction = prompt_config["CALIBRATION_INSTRUCTION"]
        if self.prompting_strategy not in [0, 1, 2, 3]:
            raise ValueError("Prompting strategy is not supported")
        task = self.dataset_info['task']
        self.prompt = prompt_template[task][self.prompting_strategy]
        self.calibration_prompt = (
            calibration_instruction.get(task, {}).get(self.prompting_strategy, None)
        )
    def get_dataset_config(self) -> None:
        """
        Loads the dataset configuration and sets up the training and testing datasets.
        """
        self.dataset_info = get_dataset_list(
            dataset_names=[self.datasets['name']],
            dataset_dir=os.path.join(self.args.config_dir, self.args.lang),
        )[0]
        self.datasets['training'], self.datasets['testing'] = load_a_dataset()
    def get_dataset_testing(self) -> Any:
        """
        Returns the testing dataset if available.
        """
        if self.datasets['testing'] is None:
            raise ValueError("Testing dataset is not available")
        return self.datasets['testing']
    def get_dataset_training(self) -> Any:
        """
        Returns the training dataset if available.
        
        Raises:
            ValueError: If the training dataset is not available.
        
        Returns:
            Any: The training dataset.
        """
        if self.datasets['training'] is None:
            raise ValueError("Training dataset is not available")
        return self.datasets['training']
    
