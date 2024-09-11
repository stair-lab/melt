"""
This module implements a pipeline for multiple choice sentiment analysis.

It includes classes for configuring the pipeline, wrapping datasets,
and managing batch and result contexts.
"""

from typing import List, Dict, Any, Callable, NamedTuple
from dataclasses import dataclass
import random

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        """Simple replacement for tqdm if it's not installed."""
        return iterable

from utils.utils import format_fewshot, unique

@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    task_name: str
    few_shot: bool
    continue_infer_data: Dict[str, List]

@dataclass
class DatasetWrapper:
    """Wrapper for dataset information and prompts."""
    dataset_info: Any
    dataset_training: Any
    prompt: Dict[str, str]
    calibration_prompt: Dict[str, str]

class BatchContext(NamedTuple):
    """Context for batch processing."""
    ds_wrapper: DatasetWrapper
    original_few_shot: List
    calib_few_shot: List
    num_choice: int

class ResultContext(NamedTuple):
    """Context for storing results."""
    data: Dict[str, List]
    selected_sample: List
    ds_wrapper: DatasetWrapper

class MultipleChoiceSentimentPipeline:
    """Pipeline for multiple choice sentiment analysis."""

    def __init__(self, config: PipelineConfig, metric_pipeline: Any, infer_pipeline: Any):
        self.config = config
        self.metric_pipeline = metric_pipeline
        self.infer_pipeline = infer_pipeline

    def run(self, ds_wrapper: DatasetWrapper, ds_loader: Any,
            saving_fn: Callable, start_idx: int = 0) -> None:
        """Run the multiple choice sentiment pipeline."""
        data = self._initialize_data()
        num_choice = len(ds_wrapper.dataset_info.label)
        if self.config.few_shot:
            selected_sample,original_few_shot,calib_few_shot=self._prepare_few_shot_data(ds_wrapper)
        else:
            selected_sample, original_few_shot, calib_few_shot = [], [], []
        batch_context = BatchContext(ds_wrapper, original_few_shot,
                                     calib_few_shot, num_choice)
        result_context = ResultContext(data, selected_sample, ds_wrapper)

        for idx, batch in enumerate(tqdm(ds_loader)):
            if idx < start_idx:
                continue

            self._process_batch(batch, batch_context, data)

            if (idx + 1) % 100 == 0:
                self._save_intermediate_results(idx + 1, result_context, saving_fn)

        self._save_final_results(result_context, saving_fn)

    def analyze_results(self, result_context: ResultContext) -> Dict[str, Any]:
        """Analyze the results of the pipeline."""
        generations = {**result_context.data, "fewshot": result_context.selected_sample}
        mean_result = self._calculate_mean_result(generations, result_context.ds_wrapper)
        std_result = self._calculate_std_result(generations, result_context.ds_wrapper)
        return {"mean": mean_result, "std": std_result}

    def _initialize_data(self) -> Dict[str, List]:
        data = {
            "predictions": [],
            "references": [],
            "generation_probs": [],
            "option_probs": []
        }
        if self.config.continue_infer_data:
            for key, value in self.config.continue_infer_data.items():
                data[key].extend(value)
        return data

    def _prepare_few_shot_data(self, ds_wrapper: DatasetWrapper) -> tuple:
        def preprocessing_a_record(rec):
            return [
                rec[ds_wrapper.dataset_info.query],
                rec[ds_wrapper.dataset_info.answer],
            ]

        classes = unique(ds_wrapper.dataset_training[ds_wrapper.dataset_info.answer])
        selected_sample = []
        for class_label in classes:
            cl_samples = ds_wrapper.dataset_training.filter(
                lambda r, label=class_label: r[ds_wrapper.dataset_info.answer] == label
            )
            selected_sample.append(
                preprocessing_a_record(
                    cl_samples[random.randint(0, len(cl_samples) - 1)]
                )
            )
        original_few_shot = format_fewshot(
            selected_sample,
            query_format=ds_wrapper.prompt["prompt"],
            answer_format=ds_wrapper.prompt["answer_format"],
        )
        calib_few_shot = format_fewshot(
            selected_sample,
            query_format=ds_wrapper.calibration_prompt["prompt"],
            answer_format=ds_wrapper.prompt["answer_format"],
        )
        return selected_sample, original_few_shot, calib_few_shot

    def _process_batch(self, batch: Dict[str, Any], batch_context: BatchContext,
                       data: Dict[str, List]) -> None:
        prompts = self._create_prompts(batch, batch_context.ds_wrapper,
                                       batch_context.original_few_shot)
        calib_prompts = self._create_calib_prompts(batch, batch_context.ds_wrapper,
                                                   batch_context.calib_few_shot)

        results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
        option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
            calib_prompts * batch_context.num_choice,
            [batch_context.ds_wrapper.dataset_info.label[choice]
             for choice in range(batch_context.num_choice)
             for _ in range(len(prompts))],
        )

        data["predictions"].extend(results)
        data["references"].extend([x.item() for x in
         batch[batch_context.ds_wrapper.dataset_info.answer]])
        data["generation_probs"].extend(logprobs)
        data["option_probs"].extend(
            [[option_logprobs[i + opt * len(prompts)]
              for opt in range(batch_context.num_choice)]
             for i in range(len(prompts))]
        )

    def _create_prompts(self, batch: Dict[str, Any], ds_wrapper: DatasetWrapper,
                        original_few_shot: List) -> List[List[Dict[str, str]]]:
        return [
            [
                {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
                *original_few_shot,
                {"role": "user", "content": ds_wrapper.prompt["prompt"].format(c)},
            ]
            for c in batch[ds_wrapper.dataset_info.query]
        ]

    def _create_calib_prompts(self, batch: Dict[str, Any], ds_wrapper: DatasetWrapper,
                              calib_few_shot: List) -> List[List[Dict[str, str]]]:
        return [
            [
                {"role": "system", "content": ds_wrapper.calibration_prompt["system_prompt"]},
                *calib_few_shot,
                {"role": "user", "content": ds_wrapper.calibration_prompt["prompt"].format(c)},
            ]
            for c in batch[ds_wrapper.dataset_info.query]
        ]

    def _save_intermediate_results(self, idx: int, result_context: ResultContext,
                                   saving_fn: Callable) -> None:
        print(f"Saving results of {idx} batches")
        generations = {**result_context.data, "fewshot": result_context.selected_sample}
        saving_fn(generations)
        mean_result = self._calculate_mean_result(generations, result_context.ds_wrapper)
        print(f"Results of {idx} batches: ", mean_result)

    def _save_final_results(self, result_context: ResultContext, saving_fn: Callable) -> None:
        generations = {**result_context.data, "fewshot": result_context.selected_sample}
        final_result = self.analyze_results(result_context)
        saving_fn(generations, final_result)

    def _calculate_mean_result(self, generations: Dict[str, Any],
                               ds_wrapper: DatasetWrapper) -> Dict[str, Any]:
        return self.metric_pipeline.run_mean(
            generations,
            self.config.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )

    def _calculate_std_result(self, generations: Dict[str, Any],
                              ds_wrapper: DatasetWrapper) -> Dict[str, Any]:
        return self.metric_pipeline.run_std(
            generations,
            self.config.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
