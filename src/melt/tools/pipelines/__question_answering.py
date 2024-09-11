"""
Module for question answering pipeline.
"""

import random
from dataclasses import dataclass
from utils.utils import format_fewshot
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class PipelineConfig:
    """
    Configuration for the question answering pipeline.
    """
    num_fs: int
    task_name: str
    config: dict

@dataclass
class Results:
    """
    Results and metrics for question answering.
    """
    predictions: list
    references: list
    generation_probs: list
    fewshot: list

@dataclass
class Context:
    """
    Context for processing batches in the question answering pipeline.
    """
    ds_wrapper: any
    pipeline_config: PipelineConfig
    metric_pipeline: any
    saving_fn: callable

def preprocess_sample(ds_wrapper, num_fs):
    """
    Preprocess and select few-shot samples from the dataset.
    """
    def preprocessing_a_record(rec):
        return [
            rec[ds_wrapper.dataset_info.context],
            rec[ds_wrapper.dataset_info.query],
            rec[ds_wrapper.dataset_info.answer]["text"][0],
        ]

    selected_sample_idx = random.sample(range(len(ds_wrapper.dataset_training)), num_fs)
    selected_sample = [
        preprocessing_a_record(ds_wrapper.dataset_training[s])
        for s in selected_sample_idx
    ]
    formatted_fewshot = format_fewshot(
        selected_sample,
        query_format=ds_wrapper.prompt["prompt"],
        answer_format=ds_wrapper.prompt["answer_format"],
    )
    return formatted_fewshot, selected_sample

def process_batch_prompts(batch, ds_wrapper, fewshot):
    """
    Create prompts for a batch of data.
    """
    return [
        [
            {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
            *fewshot,
            {"role": "user", "content": ds_wrapper.prompt["prompt"].format(c, q)},
        ]
        for c, q in zip(batch[ds_wrapper.dataset_info.context],batch[ds_wrapper.dataset_info.query])
    ]

def update_results(results, predictions_data, logprobs, batch_answers):
    """
    Update results with new data.
    """
    results.predictions.extend(predictions_data)
    results.references.extend(batch_answers)
    results.generation_probs.extend(logprobs)

def save_results_and_print_metrics(context, results, idx):
    """
    Save results and print metrics.
    """
    print(f"Saving results of {idx} batches")
    context.saving_fn(results.__dict__)
    mean_result = context.metric_pipeline.run_mean(
        results.__dict__,
        context.pipeline_config.task_name,
        context.ds_wrapper.prompt["answer_key"],
        context.ds_wrapper.dataset_info.label,
        context.pipeline_config.config
    )
    print(f"Results of {idx} batches: ", mean_result)

def __question_answering(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
    """
    Main function to perform question answering.
    """
    results = Results(
        predictions=[],
        references=[],
        generation_probs=[],
        fewshot=[]
    )

    if self.continue_infer_data:
        results.predictions = self.continue_infer_data["predictions"]
        results.references = self.continue_infer_data["references"]
        results.generation_probs = self.continue_infer_data["generation_probs"]

    if self.few_shot:
        results.fewshot, _ = preprocess_sample(ds_wrapper, self.config.num_fs)

    context = Context(
        ds_wrapper=ds_wrapper,
        pipeline_config=PipelineConfig(
            num_fs=self.config.num_fs,
            task_name=self.task_name,
            config=self.config
        ),
        metric_pipeline=self.metric_pipeline,
        saving_fn=saving_fn
    )

    idx = 0
    for batch in tqdm(ds_loader):
        if idx < start_idx:
            idx += 1
            continue

        prompts = process_batch_prompts(batch, ds_wrapper, results.fewshot)
        predictions_data, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
        batch_answers = [x[0] for x in batch[ds_wrapper.dataset_info.answer]["text"]]

        update_results(results, predictions_data, logprobs, batch_answers)
        idx += 1

        if idx % 100 == 0:
            save_results_and_print_metrics(context, results, idx)

    final_result = {
        "mean": context.metric_pipeline.run_mean(
            results.__dict__,
            context.pipeline_config.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            context.pipeline_config.config
        ),
        "std": context.metric_pipeline.run_std(
            results.__dict__,
            context.pipeline_config.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            context.pipeline_config.config
        )
    }
    context.saving_fn(results.__dict__, final_result)
