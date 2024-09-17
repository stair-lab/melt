"""
This module provides functionality for evaluating and 
generating data using specified pipelines and datasets.

The `generation` function is the main entry point of this script. It performs the following tasks:
1. Initializes the seed for reproducibility.
2. Loads and processes the dataset using `DatasetWrapper`.
3. Sets up directories for saving results if they don't already exist.
4. Handles continuation of inference from a previous run if specified.
5. Creates a DataLoader for batching dataset examples.
6. Initializes the evaluation pipeline (`EvalPipeline`).
7. Runs the evaluation pipeline and saves the results to JSON files.

The script is designed to work with various configurations 
specified in the `script_args` parameter, including options for 
few-shot prompting and continuing from previous results.

Modules used:
- `os`: For file and directory operations.
- `.tools.data`: Contains `DatasetWrapper` for 
dataset management.
- `.tools.pipelines`: Contains `EvalPipeline` for 
evaluation processes.
- `.tools.utils.utils`: Provides utility functions such as 
`save_to_json`, `set_seed`, and `read_json`.
- `torch.utils.data`: For data loading with `DataLoader`.
"""
import os
from torch.utils.data import DataLoader
from .tools.data import DatasetWrapper
from .tools.pipelines import EvalPipeline
from .tools.utils.utils import save_to_json, set_seed, read_json



def generation(script_args):
    """
    Executes the data generation process based on the provided script arguments.

    This function performs the following steps:
    1. Sets the random seed for reproducibility using `set_seed`.
    2. Loads and optionally processes the dataset using `DatasetWrapper`.
    3. Constructs filenames for saving generation results and metrics based on the script arguments.
    4. Creates necessary directories for saving results if they don't already exist.
    5. Determines the starting index and results to continue 
    inference from a previous run if specified.
    6. Initializes a `DataLoader` for batching the dataset examples.
    7. Initializes an `EvalPipeline` for evaluating the data.
    8. Runs the evaluation pipeline and saves the results using the `save_results` function.
    Args:
        script_args (ScriptArguments): An object containing the configuration 
        and parameters for the data generation process.
            - seed (int): Random seed for reproducibility.
            - smoke_test (bool): Flag to indicate if a smaller subset 
            of data should be used for testing.
            - dataset_name (str): Name of the dataset.
            - model_name (str): Name of the model.
            - output_dir (str): Directory to save generation results.
            - output_eval_dir (str): Directory to save evaluation metrics.
            - continue_infer (bool): Flag to continue inference from a previous run.
            - per_device_eval_batch_size (int): Batch size for evaluation.
            - fewshot_prompting (bool): Flag for few-shot prompting.

    Returns:
        None
    """
    set_seed(script_args.seed)

    # Load dataset (you can process it here)
    dataset_wrapper = DatasetWrapper(
        args=script_args,
    )
    if script_args.smoke_test:
        n_examples = 8
        dataset_wrapper.dataset_testing = (
            dataset_wrapper.dataset_testing.select(range(n_examples))
        )
    ds_exact_name = (
        script_args.lang
        + "_"
        + dataset_wrapper.dataset_info.task
        + "_"
        + script_args.dataset_name.split("/")[-1].replace("_", "-")
        + "_"
        + script_args.model_name.split("/")[-1].replace("_","-")
        + "_"
        + script_args.prompt_type
        + "_"
        + script_args.category
        + "_"
        + f"script_args.num_fs-shot"
        + f"_pt{dataset_wrapper.prompting_strategy}"
        + f"_seed{script_args.seed}"
    )

    json_file = os.path.join(
        script_args.output_dir, f"generations_{ds_exact_name}.json"
    )
    metric_file = os.path.join(
        script_args.output_eval_dir, f"{ds_exact_name}.json"
    )

    # Save results
    if not os.path.exists(script_args.output_dir):
        os.makedirs(script_args.output_dir)
    if not os.path.exists(script_args.output_eval_dir):
        os.makedirs(script_args.output_eval_dir)

    if script_args.continue_infer:
        if os.path.exists(json_file):
            continue_results, current_batch_idx = read_json(
                json_file, script_args.per_device_eval_batch_size
            )
            start_idx = current_batch_idx
        else:
            start_idx = 0
            continue_results = None
    else:
        start_idx = 0
        continue_results = None

    dataset_loader = DataLoader(
        dataset_wrapper.get_dataset_testing(),
        batch_size=script_args.per_device_eval_batch_size,
        shuffle=False,
    )

    # Initialize pipeline
    eval_pipeline = EvalPipeline(
        task=dataset_wrapper.dataset_info.task, config=script_args
    )

    # Evaluate
    def save_results(generations, metrics=None):
        save_to_json(generations, json_file)
        if metrics is not None:
            save_to_json(metrics, metric_file)

    eval_pipeline.run(
        ds_wrapper=dataset_wrapper,
        ds_loader=dataset_loader,
        generation_results_file=ds_exact_name,
        saving_fn=save_results,
        start_idx=start_idx,
        few_shot=script_args.fewshot_prompting,  # few-shot prompting
        continue_infer=continue_results,
    )
