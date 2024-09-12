"""
Module for multiple choice text classification using a pipeline approach.
"""

import ast
from typing import Callable, List, Dict, Any
import random
from dataclasses import dataclass
from utils.utils import format_fewshot, unique


def tqdm_fallback(iterable):
    """Fallback for tqdm if it's not installed."""
    return iterable


try:
    from tqdm import tqdm
except ImportError:
    tqdm = tqdm_fallback


@dataclass
class ClassificationConfig:
    """Configuration for the classification task."""
    task_name: str
    few_shot: bool = False
    continue_infer_data: Dict[str, List[Any]] = None


@dataclass
class SaveResultsParams:
    """Parameters for saving classification results."""
    data: Any
    ds_wrapper: Any
    saving_fn: Callable
    is_final: bool


class MultipleChoiceTextClassification:
    """
    A class for performing multiple choice text classification tasks.
    """

    def __init__(
        self,
        config: ClassificationConfig,
        metric_pipeline: Any,
        infer_pipeline: Any,
    ):
        """Initialize the MultipleChoiceTextClassification instance."""
        self.config = config
        self.metric_pipeline = metric_pipeline
        self.infer_pipeline = infer_pipeline
        self.ds_wrapper = None

    def multiple_choice_text_classification(
        self,
        ds_wrapper: Any,
        ds_loader: Any,
        saving_fn: Callable,
        start_idx: int = 0
    ) -> None:
        """
        Perform the classification task.
        """
        self.ds_wrapper = ds_wrapper
        data = self.ClassificationData(self.config.continue_infer_data)

        num_choice = len(ds_wrapper.dataset_info.label)
        few_shot_data = self.prepare_few_shot(ds_wrapper) if self.config.few_shot else None

        idx = start_idx - 1
        for idx, batch in enumerate(tqdm(ds_loader), start=start_idx):
            if idx < start_idx:
                continue

            self.process_batch(batch, data, num_choice, few_shot_data)

            if idx % 100 == 0:
                self.save_results(idx, SaveResultsParams(data, ds_wrapper, saving_fn, False))

        self.save_results(idx, SaveResultsParams(data, ds_wrapper, saving_fn, True))

    def process_batch(self, batch, data, num_choice, few_shot_data):
        """Process a single batch of data."""
        prompts = self.create_prompts(batch, self.ds_wrapper, few_shot_data)
        calib_prompts = self.create_calib_prompts(batch, self.ds_wrapper, few_shot_data)

        results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
        option_logprobs = self.compute_option_logprobs(calib_prompts, num_choice, prompts)

        data.update(results, self.process_references(batch, self.ds_wrapper), logprobs,
                    self.process_option_probs(option_logprobs, num_choice, prompts))

    def prepare_few_shot(self, ds_wrapper: Any) -> Dict[str, Any]:
        """Prepare few-shot examples for the classification task."""
        def preprocessing_a_record(rec):
            return [
                rec[ds_wrapper.dataset_info.query],
                rec[ds_wrapper.dataset_info.answer],
            ]

        classes = unique(ds_wrapper.dataset_training[ds_wrapper.dataset_info.answer])
        selected_sample = []

        for class_label in classes:
            cl_samples = ds_wrapper.dataset_training.filter(
                lambda r, label=class_label: (r[ds_wrapper.dataset_info.answer] == label)
            )
            selected_sample.append(cl_samples[random.randint(0, len(cl_samples) - 1)])

        selected_sample = [preprocessing_a_record(x) for x in selected_sample]

        return {
            "original": format_fewshot(
                selected_sample,
                query_format=ds_wrapper.prompt["prompt"],
                answer_format=ds_wrapper.prompt["answer_format"],
            ),
            "calib": format_fewshot(
                selected_sample,
                query_format=ds_wrapper.calibration_prompt["prompt"],
                answer_format=ds_wrapper.prompt["answer_format"],
            ),
            "selected_sample": selected_sample
        }

    @staticmethod
    def create_prompts(batch: Any, ds_wrapper: Any, few_shot_data:
            Dict[str, Any]) -> List[List[Dict[str, str]]]:
        """Create prompts for the classification task."""
        original_few_shot = few_shot_data["original"] if few_shot_data else []
        return [
            [
                {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
                *original_few_shot,
                {"role": "user", "content": ds_wrapper.prompt["prompt"].format(c)},
            ]
            for c in batch[ds_wrapper.dataset_info.query]
        ]

    @staticmethod
    def create_calib_prompts(
        batch: Any, ds_wrapper: Any, few_shot_data: Dict[str, Any]
    ) -> List[List[Dict[str, str]]]:
        """Create calibration prompts for the classification task."""
        calib_few_shot = few_shot_data["calib"] if few_shot_data else []
        return [
            [
                {"role": "system", "content": ds_wrapper.calibration_prompt["system_prompt"]},
                *calib_few_shot,
                {"role": "user", "content": ds_wrapper.calibration_prompt["prompt"].format(c)},
            ]
            for c in batch[ds_wrapper.dataset_info.query]
        ]

    def compute_option_logprobs(
        self, calib_prompts: List[List[Dict[str, str]]],
            num_choice: int, prompts: List[List[Dict[str, str]]]
    ) -> List[float]:
        """Compute log probabilities for each option."""
        option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
            calib_prompts * num_choice,
            [
                self.ds_wrapper.dataset_info.label[choice]
                for choice in range(num_choice)
                for _ in range(len(prompts))
            ],
        )
        return option_logprobs

    @staticmethod
    def process_references(batch: Any, ds_wrapper: Any) -> List[Any]:
        """Process references from the batch."""
        return [
            ast.literal_eval(x) if isinstance(x, str) else x.item()
            for x in batch[ds_wrapper.dataset_info.answer]
        ]

    @staticmethod
    def process_option_probs(
        option_logprobs: List[float], num_choice: int, prompts: List[List[Dict[str, str]]]
    ) -> List[List[float]]:
        """Process option probabilities."""
        return [
            [option_logprobs[i + opt * len(prompts)] for opt in range(num_choice)]
            for i in range(len(prompts))
        ]

    def save_results(self, idx: int, params: SaveResultsParams) -> None:
        """Save classification results."""
        print(f"Saving {'final' if params.is_final else 'intermediate'} results of {idx} batches")
        generations = params.data.to_dict()
        params.saving_fn(generations)

        mean_result = self.metric_pipeline.run_mean(
            generations,
            self.config.task_name,
            params.ds_wrapper.prompt["answer_key"],
            params.ds_wrapper.dataset_info.label,
            self.config.__dict__,
        )
        print(f"Results of {idx} batches: ", mean_result)

        if params.is_final:
            std_result = self.metric_pipeline.run_std(
                generations,
                self.config.task_name,
                params.ds_wrapper.prompt["answer_key"],
                params.ds_wrapper.dataset_info.label,
                self.config.__dict__,
            )
            final_result = {"mean": mean_result, "std": std_result}
            params.saving_fn(generations, final_result)

    class ClassificationData:
        """Class to manage classification data."""

        def __init__(self, continue_infer_data: Dict[str, List[Any]] = None):
            """Initialize ClassificationData."""
            if continue_infer_data:
                self.predictions = continue_infer_data["predictions"]
                self.references = continue_infer_data["references"]
                self.generation_probs = continue_infer_data["generation_probs"]
                self.option_probs = continue_infer_data["option_probs"]
            else:
                self.predictions = []
                self.references = []
                self.generation_probs = []
                self.option_probs = []

        def update(self, predictions: List[Any], references: List[Any],
                   generation_probs: List[float], option_probs: List[List[float]]) -> None:
            """Update the classification data with new batch results."""
            self.predictions.extend(predictions)
            self.references.extend(references)
            self.generation_probs.extend(generation_probs)
            self.option_probs.extend(option_probs)

        def to_dict(self) -> Dict[str, List[Any]]:
            """Convert ClassificationData to a dictionary."""
            return {
                "predictions": self.predictions,
                "references": self.references,
                "generation_probs": self.generation_probs,
                "option_probs": self.option_probs,
            }
