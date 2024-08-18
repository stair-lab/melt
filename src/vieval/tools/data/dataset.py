import os
import json
from .loader import load_a_dataset
from .parser import get_dataset_list


def eval_keys(keys):
    def eval_x(x):
        if isinstance(keys, str):
            x[keys] = eval(x[keys])
        elif isinstance(keys, list):
            for key in keys:
                x[key] = eval(x[key])
        return x

    return eval_x


class DatasetWrapper:
    def __init__(self, args) -> None:
        self.dataset_name = args.dataset_name

        self.dataset_info = None
        self.dataset_training = None
        self.dataset_testing = None

        self.args = args
        self.get_dataset_config()
        self.prompting_strategy = self.dataset_info.prompting_strategy
        self.get_prompt()

    def get_prompt(self):
        with open(
            os.path.join(
                self.args.config_dir, self.args.lang, "prompt_template.json"
            ),
            "r",
        ) as f:
            prompt_config = json.load(f)
        PROMPT_TEMPLATE = prompt_config["PROMPT_TEMPLATE"]
        CALIBRATION_INSTRUCTION = prompt_config["CALIBRATION_INSTRUCTION"]

        if self.prompting_strategy not in [0, 1, 2, 3]:
            raise ValueError("Prompting strategy is not supported")
        task = self.dataset_info.task
        self.prompt = PROMPT_TEMPLATE[task][self.prompting_strategy]
        if task in CALIBRATION_INSTRUCTION:
            self.calibration_prompt = CALIBRATION_INSTRUCTION[task][
                self.prompting_strategy
            ]
        else:
            self.calibration_prompt = None

    def get_dataset_config(self):
        self.dataset_info = get_dataset_list(
            dataset_names=[self.dataset_name],
            dataset_dir=os.path.join(self.args.config_dir, self.args.lang),
        )[0]
        self.dataset_training, self.dataset_testing = load_a_dataset(
            self.dataset_info, self.args
        )

    def get_dataset_testing(self):
        if self.dataset_testing is None:
            raise ValueError("Dataset testing is not available")
        return self.dataset_testing

    def get_dataset_training(self):
        if self.dataset_training is None:
            raise ValueError("Dataset training is not available")
        return self.dataset_training
