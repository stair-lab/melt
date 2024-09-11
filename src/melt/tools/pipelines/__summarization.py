"""
This module contains the summarization pipeline for processing and evaluating
text summarization tasks.

It uses few-shot learning for prompt generation and handles the inference process
using the provided model. Results are saved periodically and at the end.
"""

import random
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from utils.utils import format_fewshot

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        """
        A simple replacement for tqdm if it's not installed.
        
        Args:
            iterable: The iterable to wrap.
        
        Returns:
            The original iterable.
        """
        return iterable

@dataclass
class SummarizationConfig:
    """Configuration for the summarization pipeline."""
    num_fs: int
    few_shot: bool
    continue_infer_data: Dict[str, List] = None

class SummarizationPipeline:
    """
    A pipeline for summarizing documents and evaluating the performance.
    
    This class encapsulates the logic for document summarization, including
    few-shot learning, batch processing, and result evaluation.
    """

    def __init__(self, config: SummarizationConfig, metric_pipeline:
         Any, infer_pipeline: Any, task_name: str):
        self.config = config
        self.metric_pipeline = metric_pipeline
        self.infer_pipeline = infer_pipeline
        self.task_name = task_name
        self.data = self._initialize_data()

    def run_summarization(self, ds_wrapper: Any, ds_loader:
         Any, saving_fn: Callable, start_idx: int = 0) -> None:
        """
        Run the summarization pipeline.

        Args:
            ds_wrapper: A wrapper for the dataset, providing information and prompts.
            ds_loader: DataLoader for loading batches of data.
            saving_fn: Function to save the results.
            start_idx: Index to start processing from.
        """
        selected_sample, original_few_shot = self._prepare_few_shot_data(ds_wrapper)

        for idx, batch in enumerate(tqdm(ds_loader)):
            if idx < start_idx:
                continue

            self._process_batch(batch, ds_wrapper, original_few_shot)

            if (idx + 1) % 100 == 0:
                self._save_intermediate_results(idx + 1, selected_sample, saving_fn, ds_wrapper)

        self._save_final_results(selected_sample, saving_fn, ds_wrapper)

    def get_results(self) -> Dict[str, List]:
        """
        Get the current results of the summarization pipeline.

        Returns:
            A dictionary containing the current results.
        """
        return self.data

    def _initialize_data(self) -> Dict[str, List]:
        """Initialize data structures for storing results."""
        data = {
            "original_documents": [],
            "predictions": [],
            "references": [],
            "generation_probs": []
        }
        if self.config.continue_infer_data:
            for key, value in self.config.continue_infer_data.items():
                data[key].extend(value)
        return data

    def _prepare_few_shot_data(self, ds_wrapper: Any) -> tuple:
        """Prepare few-shot samples and format them."""
        if not self.config.few_shot:
            return [], []

        selected_sample = self._select_few_shot_samples(ds_wrapper)
        original_few_shot = format_fewshot(
            selected_sample,
            query_format=ds_wrapper.prompt["prompt"],
            answer_format=ds_wrapper.prompt["answer_format"],
        )
        return selected_sample, original_few_shot

    def _select_few_shot_samples(self, ds_wrapper: Any) -> List[List[str]]:
        """Select few-shot samples from the training dataset."""
        selected_sample_idx = random.sample(
            range(len(ds_wrapper.dataset_training)), self.config.num_fs
        )
        return [
            [
                ds_wrapper.dataset_training[s][ds_wrapper.dataset_info.source],
                ds_wrapper.dataset_training[s][ds_wrapper.dataset_info.target]
            ]
            for s in selected_sample_idx
        ]
    def _process_batch(self, batch: Dict[str, Any], ds_wrapper: Any,
         original_few_shot: List[Dict[str, str]]) -> None:
        """Process a single batch of data."""
        prompts = self._create_prompts(batch, ds_wrapper, original_few_shot)
        results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)

        self.data["original_documents"].extend(batch[ds_wrapper.dataset_info.source])
        self.data["predictions"].extend(results)
        self.data["references"].extend(batch[ds_wrapper.dataset_info.target])
        self.data["generation_probs"].extend(logprobs)
    def _create_prompts(self, batch: Dict[str, Any], ds_wrapper: Any,
         original_few_shot: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        """Create prompts for the current batch."""
        return [
            [
                {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
                *original_few_shot,
                {"role": "user", "content": ds_wrapper.prompt["prompt"].format(document)},
            ]
            for document in batch[ds_wrapper.dataset_info.source]
        ]
    def _save_intermediate_results(self, idx: int, selected_sample: List[List[str]],
        saving_fn: Callable, ds_wrapper: Any) -> None:
        """Save intermediate results and print mean results."""
        print(f"Saving results of {idx} batches")
        generations = {**self.data, "fewshot": selected_sample}
        saving_fn(generations)
        mean_result = self._calculate_mean_result(generations, ds_wrapper)
        print(f"Results of {idx} batches: ", mean_result)
    def _save_final_results(self, selected_sample: List[List[str]],
         saving_fn: Callable, ds_wrapper: Any) -> None:
        """Save final results including mean and standard deviation."""
        generations = {**self.data, "fewshot": selected_sample}
        mean_result = self._calculate_mean_result(generations, ds_wrapper)
        std_result = self._calculate_std_result(generations, ds_wrapper)
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)
    def _calculate_mean_result(self, generations: Dict[str, Any],ds_wrapper: Any) -> Dict[str, Any]:
        """Calculate mean results using the metric pipeline."""
        return self.metric_pipeline.run_mean(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )

    def _calculate_std_result(self, generations: Dict[str, Any], ds_wrapper: Any) -> Dict[str, Any]:
        """Calculate standard deviation of results using the metric pipeline."""
        return self.metric_pipeline.run_std(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
