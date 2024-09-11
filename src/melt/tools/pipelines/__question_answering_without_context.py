"""
Module for handling question answering without context. This module processes data in batches,
performs inference, and saves results, including handling few-shot learning if specified.
"""

import random
import collections  # Added import for collections
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
from utils.utils import format_fewshot  # Ensure this is used if necessary

# Define a named tuple to group related arguments
BatchProcessingArgs = collections.namedtuple('BatchProcessingArgs', [
    'ds_wrapper',
    'ds_loader',
    'results',
    'saving_fn',
    'start_idx'
])

def __question_answering_without_context(
    self, ds_wrapper, ds_loader, saving_fn, start_idx=0
):
    """
    Handles question answering without context, processes batches of data, and saves results.

    Args:
        self: The instance of the class.
        ds_wrapper: Data structure containing dataset information.
        ds_loader: Data loader for the dataset.
        saving_fn: Function to save the results.
        start_idx: Index to start processing from (default is 0).
    """
    results = initialize_results()

    if self.continue_infer_data:
        load_existing_data(self, results)

    if self.few_shot:
        handle_few_shot_learning(self, ds_wrapper, results)

    # Create a named tuple for the arguments
    args = BatchProcessingArgs(
        ds_wrapper=ds_wrapper,
        ds_loader=ds_loader,
        results=results,
        saving_fn=saving_fn,
        start_idx=start_idx
    )

    process_batches(self, args)

def process_batches(self, args):
    """
    Processes batches of data, updates results, and saves them.

    Args:
        self: The instance of the class.
        args: A named tuple containing:
            - ds_wrapper: Data structure containing dataset information.
            - ds_loader: Data loader for the dataset.
            - results: Dictionary containing results.
            - saving_fn: Function to save the results.
            - start_idx: Index to start processing from.
    """
    for idx, batch in enumerate(tqdm(args.ds_loader), start=0):
        if idx < args.start_idx:
            continue

        prompts, calib_prompts = create_prompts(args.ds_wrapper, batch, args.results)

        infer_results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
        calibprob_batch, _ = self.infer_pipeline.compute_logprob_and_length(
            calib_prompts, batch[args.ds_wrapper.dataset_info.answer]
        )

        update_results(args.results, infer_results, batch, logprobs, calibprob_batch)

        if (idx + 1) % 100 == 0:
            save_intermediate_results(self, idx, args.results, args.saving_fn, args.ds_wrapper)

    save_final_results(self, args.results, args.saving_fn, args.ds_wrapper)

def initialize_results():
    """
    Initializes the results dictionary for storing inference data.

    Returns:
        dict: Dictionary containing lists for storing predictions, references, probabilities, etc.
    """
    return {
        "predictions": [],
        "references": [],
        "generation_probs": [],
        "calibration_probs": [],
        "fewshot": []
    }

def load_existing_data(self, results):
    """
    Loads existing inference data if available and extends the results dictionary.

    Args:
        self: The instance of the class.
        results: Dictionary containing results.
    """
    for key, value in self.continue_infer_data.items():
        if key in results:
            results[key].extend(value)

def handle_few_shot_learning(self, ds_wrapper, results):
    """
    Handles few-shot learning by selecting samples and formatting prompts.

    Args:
        self: The instance of the class.
        ds_wrapper: Data structure containing dataset information.
        results: Dictionary containing results.
    """
    selected_sample_idx = random.sample(
        range(len(ds_wrapper.dataset_training)), self.config.num_fs
    )
    selected_sample = [
        [rec[ds_wrapper.dataset_info.query], rec[ds_wrapper.dataset_info.answer]]
        for s in selected_sample_idx
        if (rec := ds_wrapper.dataset_training[s])
    ]

    results["fewshot"] = selected_sample
    results["original_few_shot"] = format_fewshot(
        selected_sample,
        query_format=ds_wrapper.prompt["prompt"],
        answer_format=ds_wrapper.prompt["answer_format"]
    )
    results["calib_few_shot"] = format_fewshot(
        selected_sample,
        query_format=ds_wrapper.calibration_prompt["prompt"],
        answer_format=ds_wrapper.prompt["answer_format"]
    )

def create_prompts(ds_wrapper, batch, results):
    """
    Creates prompts for inference based on the dataset and results.

    Args:
        ds_wrapper: Data structure containing dataset information.
        batch: Batch of data to process.
        results: Dictionary containing results.

    Returns:
        tuple: Prompts and calibration prompts.
    """
    prompts = [
        [
            {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
            *results.get("original_few_shot", []),
            {"role": "user", "content": ds_wrapper.prompt["prompt"].format(q)}
        ]
        for q in batch[ds_wrapper.dataset_info.query]
    ]

    calib_prompts = [
        [
            {"role": "system", "content": ds_wrapper.calibration_prompt["system_prompt"]},
            *results.get("calib_few_shot", []),
            {"role": "user", "content": ds_wrapper.calibration_prompt["prompt"].format(q)}
        ]
        for q in batch[ds_wrapper.dataset_info.query]
    ]

    return prompts, calib_prompts

def update_results(results, infer_results, batch, logprobs, calibprob_batch):
    """
    Updates the results dictionary with new inference data.

    Args:
        results: Dictionary containing results.
        infer_results: List of inference results.
        batch: Batch of data.
        logprobs: List of generation probabilities.
        calibprob_batch: List of calibration probabilities.
    """
    results["predictions"].extend(infer_results)
    results["references"].extend(batch[results.ds_wrapper.dataset_info.answer])
    results["generation_probs"].extend(logprobs)
    results["calibration_probs"].extend(calibprob_batch)

def save_intermediate_results(self, idx, results, saving_fn, ds_wrapper):
    """
    Saves intermediate results after processing a batch of data.

    Args:
        self: The instance of the class.
        idx: Index of the current batch.
        results: Dictionary containing results.
        saving_fn: Function to save the results.
        ds_wrapper: Data structure containing dataset information.
    """
    print(f"Saving results of {idx + 1} batches")
    mean_result = self.metric_pipeline.run_mean(
        results,
        self.task_name,
        ds_wrapper.prompt["answer_key"],
        ds_wrapper.dataset_info.label,
        self.config
    )
    print(f"Results of {idx + 1} batches: ", mean_result)
    saving_fn(results)

def save_final_results(self, results, saving_fn, ds_wrapper):
    """
    Saves the final results after all batches have been processed.

    Args:
        self: The instance of the class.
        results: Dictionary containing results.
        saving_fn: Function to save the results.
        ds_wrapper: Data structure containing dataset information.
    """
    mean_result = self.metric_pipeline.run_mean(
        results,
        self.task_name,
        ds_wrapper.prompt["answer_key"],
        ds_wrapper.dataset_info.label,
        self.config
    )
    std_result = self.metric_pipeline.run_std(
        results,
        self.task_name,
        ds_wrapper.prompt["answer_key"],
        ds_wrapper.dataset_info.label,
        self.config
    )
    final_result = {"mean": mean_result, "std": std_result}
    saving_fn(results, final_result)

