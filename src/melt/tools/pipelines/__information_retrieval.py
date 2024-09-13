"information_retrieval"
import random
from typing import List

from dataclasses import dataclass
from tqdm import tqdm
from utils.utils import format_fewshot, column

@dataclass
class PromptCreationConfig:
    "Class"
    system_prompt: str
    few_shot: List[dict]
    prompt_format: str
    batch_passage_size: int
    top30_passages: List[str]
    query: str = None

@dataclass
class SavePromptConfig:
    "Class"
    results: list
    logprobs: list
    top30_passages: list
    ds_wrapper: object
    ref_passage_id: str

@dataclass
class BatchProcessingParams:
    "Class"
    batch: dict
    ds_wrapper: object
    original_few_shot: list
    calib_few_shot: list
    batch_passage_size: int
    self: object

@dataclass
class InformationRetrievalConfig:
    "Class"
    ds_wrapper: object
    ds_loader: object
    saving_fn: callable
    start_idx: int
    batch_passage_size: int
    self: object

@dataclass
class InformationRetrievalParams:
    "Class"
    ds_wrapper: object
    ds_loader: object
    saving_fn: callable
    start_idx: int
    batch_passage_size: int
    self: object

@dataclass
class FinalSavingMetricsParams:
    "Class"
    predictions: list
    selected_sample: list
    saving_fn: callable
    self: object
    ds_wrapper: object

def preprocess_record(rec, ds_wrapper):
    """Preprocess a record to extract passages, query, and answer."""
    return [
        rec[ds_wrapper.dataset_info.passages],
        rec[ds_wrapper.dataset_info.query],
        rec[ds_wrapper.dataset_info.answer],
    ]

def create_fewshot_samples(ds_wrapper):
    """Create fewshot samples for training and calibration."""
    random_sample = list(random.sample(list(ds_wrapper.dataset_training), 1))[0]
    first_sample = {
        "passages": random_sample["positive"],
        "query": random_sample[ds_wrapper.dataset_info.query],
        "references": ds_wrapper.dataset_info.label[0],
    }
    second_sample = {
        "passages": random_sample["negative"],
        "query": random_sample[ds_wrapper.dataset_info.query],
        "references": ds_wrapper.dataset_info.label[1],
    }
    selected_sample = [
        preprocess_record(s, ds_wrapper)
        for s in [first_sample, second_sample]
    ]
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
    return original_few_shot, calib_few_shot, selected_sample

def generate_batch_prompts(batch, ds_wrapper, config: PromptCreationConfig):
    """Generate prompts and calibration prompts for the given batch."""
    passages = batch[ds_wrapper.dataset_info.passages]
    prompts, calib_prompts = [], []

    for i in range(len(batch[ds_wrapper.dataset_info.type_id])):
        query = batch[ds_wrapper.dataset_info.query][i]
        top30_passages = column(passages["passage"], i)

        prompt_config = PromptCreationConfig(
            system_prompt=config.system_prompt,
            few_shot=config.few_shot,
            prompt_format=config.prompt_format,
            batch_passage_size=config.batch_passage_size,
            top30_passages=top30_passages,
            query=query
        )

        prompts.extend(create_prompts(prompt_config))
        calib_prompts.extend(create_prompts(
            PromptCreationConfig(
                system_prompt=config.system_prompt,
                few_shot=config.calib_few_shot,
                prompt_format=config.prompt_format,
                batch_passage_size=config.batch_passage_size,
                top30_passages=top30_passages,
                query=query
            )
        ))

    return prompts, calib_prompts


def create_prompts(config: PromptCreationConfig) -> List[List[dict]]:
    """Create prompts for a batch of passages."""
    if config.query is None:
        config.query = "default_query_value"  # Or compute from other arguments

    return [
        [
            {"role": "system", "content": config.system_prompt},
            *config.few_shot,
            {"role": "user", "content": config.prompt_format.format(p, config.query)},
        ]
        for start in range(0, len(config.top30_passages), config.batch_passage_size)
        for p in config.top30_passages[start:start + config.batch_passage_size]
    ]

def generate_save_each_prompt(config: SavePromptConfig):
    """Generate the final data structure for saving each prompt's results."""
    return [
        {
            "query_id": query_id,
            "query": query,
            "passage_id": psg_id,
            "passage": passage,
            "label": int(psg_id == config.ref_passage_id),
            "prediction": result,
            "generation_probs": prob,
            "calib_probs": calib_prob
        }
        for result, prob, psg_id, passage, query_id, query, calib_prob in zip(
            config.results,
            config.logprobs,
            column(config.top30_passages, 0),
            config.top30_passages,
            range(len(config.top30_passages)),
            [config.ds_wrapper.dataset_info.query] * len(config.top30_passages),
            [0] * len(config.top30_passages)  # Placeholder for calibration probabilities
        )
    ]

def process_batch(params: BatchProcessingParams):
    """Process a single batch of data."""
    config = PromptCreationConfig(
        top30_passages=params.ds_wrapper.dataset_info.passages,
        query=params.ds_wrapper.dataset_info.query,
        few_shot=params.original_few_shot,
        system_prompt=params.ds_wrapper.prompt["system_prompt"],
        prompt_format=params.ds_wrapper.prompt["prompt"],
        batch_passage_size=params.batch_passage_size
    )

    prompts, _ = generate_batch_prompts(params.batch, params.ds_wrapper, config)
    results, logprobs, _ = params.self.infer_pipeline(prompts, return_probs=True)
    ref_passage_id = params.batch[params.ds_wrapper.dataset_info.answer][0][0]
    top30_passages = column(params.batch[params.ds_wrapper.dataset_info.passages]["passage"], 0)

    save_config = SavePromptConfig(
        results=results,
        logprobs=logprobs,
        top30_passages=top30_passages,
        ds_wrapper=params.ds_wrapper,
        ref_passage_id=ref_passage_id
    )
    return generate_save_each_prompt(save_config)

def save_and_print_results(self, idx, predictions, selected_sample, saving_fn):
    """Save intermediate results and print metrics."""
    print(f"Saving results of {idx} batches")
    generations = {
        "fewshot": selected_sample,
        "predictions": predictions,
    }
    saving_fn(generations)
    mean_result = self.metric_pipeline.run_mean(
        generations,
        self.task_name,
        self.ds_wrapper.prompt["answer_key"],
        self.ds_wrapper.dataset_info.label,
        self.config,
        ref_dataset=self.ds_wrapper.dataset_testing,
    )
    print(f"Results of {idx} batches: ", mean_result)
    return mean_result

def final_saving_and_metrics(self, predictions, selected_sample, saving_fn):
    """Final saving and metrics calculation."""
    generations = {"fewshot": selected_sample, "predictions": predictions}
    mean_result = self.metric_pipeline.run_mean(
        generations,
        self.task_name,
        self.ds_wrapper.prompt["answer_key"],
        self.ds_wrapper.dataset_info.label,
        self.config,
        ref_dataset=self.ds_wrapper.dataset_testing,
    )
    std_result = self.metric_pipeline.run_std(
        generations,
        self.task_name,
        self.ds_wrapper.prompt["answer_key"],
        self.ds_wrapper.dataset_info.label,
        self.config,
        ref_dataset=self.ds_wrapper.dataset_testing,
    )
    final_result = {"mean": mean_result, "std": std_result}
    saving_fn(generations, final_result)

def __information_retrieval(config: InformationRetrievalConfig):
    """Main function for information retrieval."""
    predictions = []

    # Create fewshot samples
    original_few_shot, calib_few_shot, selected_sample = create_fewshot_samples(config.ds_wrapper)

    for idx, batch in enumerate(tqdm(config.ds_loader), start=0):
        if idx < config.start_idx:
            continue

        # Setup configurations
        batch_params = BatchProcessingParams(
            batch=batch,
            ds_wrapper=config.ds_wrapper,
            original_few_shot=original_few_shot,
            calib_few_shot=calib_few_shot,
            batch_passage_size=config.batch_passage_size,
            self=config.self
        )

        # Process batch
        save_each_prompt = process_batch(batch_params)
        predictions.extend(save_each_prompt)

        if idx % 100 == 0:
            config.self.save_and_print_results(idx, predictions, selected_sample, config.saving_fn)

    # Final saving
    config.self.final_saving_and_metrics(predictions, selected_sample, config.saving_fn)
