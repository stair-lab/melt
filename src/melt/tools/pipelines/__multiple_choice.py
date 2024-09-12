" __multiple_choice"
import ast
import random
from dataclasses import dataclass
from tqdm import tqdm
from utils.utils import format_fewshot
@dataclass
class DataConfig:
    " Classs"
    ds_wrapper: object
    ds_loader: object
    infer_pipeline: object
    metric_pipeline: object

@dataclass
class SaveConfig:
    "Class"
    saving_fn: callable
    continue_infer_data: dict = None

@dataclass
class ProcessorConfig:
    "Class"
    data_config: DataConfig
    save_config: SaveConfig
    task_name: str
    config: object
    few_shot: bool = False

class DataProcessor:
    """Class to handle data processing for multiple-choice tasks."""
    def __init__(self, ds_wrapper, config):
        self.ds_wrapper = ds_wrapper
        self.config = config
        self.num_choice = len(ds_wrapper.dataset_info.label)

    def format_list_ans(self, ans_list):
        """Format list of answers."""
        return "\n".join(
            f"{self.ds_wrapper.dataset_info.label[ans[0]]}: ''' {ans[1]} '''"
            for ans in enumerate(ans_list)
        )

    def preprocess_record(self, rec):
        """Preprocess a single record."""
        return [
            rec[self.ds_wrapper.dataset_info.context],
            rec[self.ds_wrapper.dataset_info.query],
            self.format_list_ans(ast.literal_eval(rec[self.ds_wrapper.dataset_info.options])),
            rec[self.ds_wrapper.dataset_info.answer],
        ]

    def prepare_few_shot(self, dataset):
        """Prepare few-shot examples."""
        selected_sample_idx = list(random.sample(range(len(dataset)), self.config.num_fs))
        selected_samples = [self.preprocess_record(dataset[s]) for s in selected_sample_idx]
        original_few_shot = format_fewshot(
            selected_samples,
            query_format=self.ds_wrapper.prompt["prompt"],
            answer_format=self.ds_wrapper.prompt["answer_format"]
        )
        calib_few_shot = format_fewshot(
            selected_samples,
            query_format=self.ds_wrapper.calibration_prompt["prompt"],
            answer_format=self.ds_wrapper.prompt["answer_format"]
        )
        return selected_samples, original_few_shot, calib_few_shot

class PromptGenerator:
    """Class to generate prompts for inference."""
    def __init__(self, ds_wrapper, original_few_shot, calib_few_shot):
        self.ds_wrapper = ds_wrapper
        self.original_few_shot = original_few_shot
        self.calib_few_shot = calib_few_shot

    def format_list_ans(self, ans_list):
        """Format list of answers."""
        return "\n".join(
            f"{self.ds_wrapper.dataset_info.label[ans[0]]}: ''' {ans[1]} '''"
            for ans in enumerate(ans_list)
        )

    def create_prompts(self, batch):
        """Create prompts for each record in the batch."""
        prompts = []
        calib_prompts = []
        remap_order_batch = []
        for context, query, options_str in zip(
            batch[self.ds_wrapper.dataset_info.context],
            batch[self.ds_wrapper.dataset_info.query],
            batch[self.ds_wrapper.dataset_info.options],
        ):
            options = ast.literal_eval(options_str)
            order_shuffle = list(range(len(options)))
            if self.ds_wrapper.dataset_info.random:
                random.shuffle(order_shuffle)
            remap_order_batch.append(order_shuffle)
            new_opts = [options[i] for i in order_shuffle]
            prompts.append([
                {"role": "system", "content": self.ds_wrapper.prompt["system_prompt"]},
                *self.original_few_shot,
                {"role": "user", "content": self.ds_wrapper.prompt["prompt"].format(
                    context, query, self.format_list_ans(new_opts)
                )},
            ])
            calib_prompts.append([
                {"role": "system", "content": self.ds_wrapper.calibration_prompt["system_prompt"]},
                *self.calib_few_shot,
                {"role": "user", "content": self.ds_wrapper.calibration_prompt["prompt"].format(
                    context, query, self.format_list_ans(new_opts)
                )},
            ])
        return prompts, calib_prompts, remap_order_batch

class Inferencer:
    """Class to handle inference and log-probability computations."""
    def __init__(self, infer_pipeline, ds_wrapper):
        self.infer_pipeline = infer_pipeline
        self.ds_wrapper = ds_wrapper

    def infer(self, prompts):
        """Perform inference on prompts."""
        return self.infer_pipeline(prompts, return_probs=True)

    def compute_logprobs(self, calib_prompts, num_choice):
        """Compute log-probabilities for the given prompts."""
        return self.infer_pipeline.compute_logprob_and_length(
            calib_prompts * num_choice,
            [self.ds_wrapper.dataset_info.label[choice] for choice in range(num_choice)
             for _ in range(len(calib_prompts))]
        )

class ResultsHandler:
    """Class to handle results and compute metrics."""
    def __init__(self, metric_pipeline, task_name, config, saving_fn):
        self.metric_pipeline = metric_pipeline
        self.task_name = task_name
        self.config = config
        self.saving_fn = saving_fn
        self.option_order_all = []
        self.selected_sample = []
        self.ds_wrapper = None  # Placeholder, set it during initialization

    def set_ds_wrapper(self, ds_wrapper):
        """Set ds_wrapper for the results handler."""        
        self.ds_wrapper = ds_wrapper

    def handle_results(self, results, logprobs, option_calib_out, remap_order_batch):
        """Handle and save the results."""
        predictions = results
        references = [
            self.ds_wrapper.dataset_info.label[
                remap.index(self.ds_wrapper.dataset_info.label.index(x))]
            for x, remap in zip(self.ds_wrapper.dataset_info.answer, remap_order_batch)
        ]
        generation_probs = logprobs
        option_probs = option_calib_out
        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "option_probs": option_probs,
            "option_orders": self.option_order_all,
            "fewshot": self.selected_sample,
        }
        self.saving_fn(generations)
        mean_result = self.metric_pipeline.run_mean(
            generations, self.task_name, self.ds_wrapper.prompt["answer_key"],
            self.ds_wrapper.dataset_info.label, self.config
        )
        std_result = self.metric_pipeline.run_std(
            generations, self.task_name, self.ds_wrapper.prompt["answer_key"],
            self.ds_wrapper.dataset_info.label, self.config
        )
        final_result = {"mean": mean_result, "std": std_result}
        self.saving_fn(generations, final_result)

    def compute_final_results(self, predictions, references, generation_probs, option_probs):
        """Compute final results based on predictions, references, and probabilities."""
        return {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "option_probs": option_probs,
            "option_orders": self.option_order_all,
            "fewshot": self.selected_sample,
        }

class MultipleChoiceProcessor:
    """Class to process multiple-choice tasks."""
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.data_processor = DataProcessor(config.data_config.ds_wrapper, config.config)
        self.prompt_generator = None
        self.inferencer = Inferencer(config.data_config.infer_pipeline,
                                      config.data_config.ds_wrapper)
        self.results_handler = ResultsHandler(
            config.data_config.metric_pipeline,
            config.task_name,
            config.config,
            config.save_config.saving_fn
        )
        self.results_handler.set_ds_wrapper(config.data_config.ds_wrapper)

    def initialize_few_shot(self):
        """Initialize few-shot examples."""
        if self.config.few_shot:
            selected_samples, original_few_shot, calib_few_shot = (
                self.data_processor.prepare_few_shot(
                    self.config.data_config.ds_wrapper.dataset_training))
            self.prompt_generator = PromptGenerator(self.config.data_config.ds_wrapper,
                                                     original_few_shot, calib_few_shot)
            self.results_handler.selected_sample = selected_samples

    def process_batch(self, batch):
        """Process a batch of data."""
        prompts, calib_prompts, remap_order_batch = self.prompt_generator.create_prompts(batch)
        results, logprobs = self.inferencer.infer(prompts)
        option_logprobs = self.inferencer.compute_logprobs(
            calib_prompts, self.data_processor.num_choice)

        opt_calib_out = [
            [option_logprobs[i + opt * len(prompts)] for opt
             in range(self.data_processor.num_choice)]
            for i in range(len(prompts))
        ]
        return results, logprobs, opt_calib_out, remap_order_batch

    def __multiple_choice(self, start_idx=0):
        """Run the processing pipeline."""
        predictions = []
        references = []
        generation_probs = []
        option_probs = []
        idx = 0
        if self.config.save_config.continue_infer_data is not None:
            predictions.extend(self.config.save_config.continue_infer_data["predictions"])
            references.extend(self.config.save_config.continue_infer_data["references"])
            generation_probs.extend(self.config.
                                    save_config.continue_infer_data["generation_probs"])
            option_probs.extend(self.config.save_config.
                                continue_infer_data["option_probs"])
            self.results_handler.option_order_all.extend(self.config.
                                                         save_config.
                                                         continue_infer_data["option_orders"])

        self.initialize_few_shot()
        for batch in tqdm(self.config.data_config.ds_loader, desc="Processing batches"):
            if idx < start_idx:
                idx += 1
                continue
            batch_results = self.process_batch(batch)
            predictions.extend(batch_results[0])
            references.extend(batch[self.config.data_config.ds_wrapper.dataset_info.answer])
            generation_probs.extend(batch_results[1])
            option_probs.extend(batch_results[2])
            self.results_handler.option_order_all.extend(batch_results[3])
            self.results_handler.handle_results(*batch_results)

        self.results_handler.handle_results(
            predictions, references, generation_probs, option_probs
        )
        return predictions, references, generation_probs, option_probs

    def run_processing_pipeline(self, start_idx=0):
        """Run the processing pipeline."""
        return self.__multiple_choice(start_idx)
