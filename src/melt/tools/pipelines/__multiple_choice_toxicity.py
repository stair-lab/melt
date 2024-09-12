"__multiple_choice_toxicity "
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Optional
import random
from tqdm import tqdm

@dataclass
class ClassificationData:
    """Data structure for classification results."""
    predictions: List[Any] = None
    references: List[Any] = None
    generation_probs: List[float] = None
    option_probs: List[List[float]] = None

    def __post_init__(self):
        self.predictions = self.predictions or []
        self.references = self.references or []
        self.generation_probs = self.generation_probs or []
        self.option_probs = self.option_probs or []

    def update(self, predictions: List[Any], references: List[Any],
               generation_probs: List[float], option_probs: List[List[float]]) -> None:
        """Update the ClassificationData with new values."""
        self.predictions.extend(predictions)
        self.references.extend(references)
        self.generation_probs.extend(generation_probs)
        self.option_probs.extend(option_probs)

    def to_dict(self) -> Dict[str, List[Any]]:
        """Convert ClassificationData to dictionary."""
        return {
            "predictions": self.predictions,
            "references": self.references,
            "generation_probs": self.generation_probs,
            "option_probs": self.option_probs,
        }

@dataclass
class BatchInfo:
    """Grouped information about batch processing."""
    batch: Any
    logprobs: List[float]
    option_logprobs: List[float]

@dataclass
class ClassificationDataUpdateParams:
    """Parameters for updating ClassificationData."""
    data: ClassificationData
    results: List[Any]
    batch_info: BatchInfo
    num_choice: int
    num_prompts: int
    ds_wrapper: Any

@dataclass
class ClassificationConfig:
    """Configuration for classification tasks."""
    task_name: str
    few_shot: bool = False
    continue_infer_data: Optional[Dict[str, List[Any]]] = None

@dataclass
class PipelineConfig:
    """Configuration for pipelines."""
    infer_pipeline: Any
    metric_pipeline: Any

@dataclass
class ClassifierConfig:
    """Grouped configuration for the classifier."""
    classification_config: ClassificationConfig
    pipeline_config: PipelineConfig

@dataclass
class BatchProcessingParams:
    """Parameters for batch processing."""
    data: ClassificationData
    batch: Any
    ds_wrapper: Any
    few_shot_data: tuple
    num_choice: int

@dataclass
class SaveResultsParams:
    """Parameters for saving results."""
    data: ClassificationData
    saving_fn: Callable
    is_final: bool
    ds_wrapper: Any

class MultipleChoiceToxicityClassifier:
    """Classifier for multiple-choice toxicity classification."""

    def __init__(self, config: ClassifierConfig):
        """Initialize the classifier."""
        self.config = config
        self._classification_data = self._initialize_classification_data()

    def classify(
        self, ds_wrapper: Any, ds_loader: Any, saving_fn: Callable, start_idx: int = 0
    ) -> None:
        """Perform classification on the given dataset."""
        num_choice = len(ds_wrapper.dataset_info.label)
        few_shot_data = (self._prepare_few_shot(ds_wrapper) if
                         self.config.classification_config.few_shot else ([], []))

        for idx, batch in enumerate(tqdm(ds_loader), start=start_idx):
            self._process_batch(BatchProcessingParams(
                self._classification_data, batch, ds_wrapper, few_shot_data, num_choice
            ))

            if idx % 100 == 0:
                self._save_intermediate_results(saving_fn, ds_wrapper)

        self._save_final_results(saving_fn, ds_wrapper)

    def get_classification_results(self) -> Dict[str, List[Any]]:
        """Retrieve the current classification results."""
        return self._classification_data.to_dict()

    # pylint: disable=W0238
    def __multiple_choice_toxicity(
        self, ds_wrapper: Any, ds_loader: Any, saving_fn: Callable, start_idx: int = 0
    ) -> None:
        """Perform classification on the given dataset."""
        num_choice = len(ds_wrapper.dataset_info.label)
        few_shot_data = (self._prepare_few_shot(ds_wrapper) if
                         self.config.classification_config.few_shot else ([], []))

        for idx, batch in enumerate(tqdm(ds_loader), start=start_idx):
            self._process_batch(BatchProcessingParams(
                self._classification_data, batch, ds_wrapper, few_shot_data, num_choice
            ))

            if idx % 100 == 0:
                self._save_intermediate_results(saving_fn, ds_wrapper)

        self._save_final_results(saving_fn, ds_wrapper)

    def _process_batch(self, params: BatchProcessingParams) -> None:
        """Process a single batch of data."""
        prompts, calib_prompts = self._create_prompts_and_calib_prompts(
            params.batch, params.ds_wrapper, params.few_shot_data
        )
        results, logprobs, _ = (
            self.config.pipeline_config.infer_pipeline(prompts, return_probs=True))
        option_logprobs = self._compute_option_logprobs(
            calib_prompts, params.num_choice, params.ds_wrapper
        )

        batch_info = (
            BatchInfo(batch=params.batch, logprobs=logprobs, option_logprobs=option_logprobs))

        self._update_classification_data(ClassificationDataUpdateParams(
            data=params.data, results=results, batch_info=batch_info,
            num_choice=params.num_choice, num_prompts=len(prompts), ds_wrapper=params.ds_wrapper
        ))

    def _initialize_classification_data(self) -> ClassificationData:
        """Initialize ClassificationData with continue inference data."""
        continue_data = self.config.classification_config.continue_infer_data or {}
        return ClassificationData(
            predictions=continue_data.get("predictions", []),
            references=continue_data.get("references", []),
            generation_probs=continue_data.get("generation_probs", []),
            option_probs=continue_data.get("option_probs", []),
        )

    def _prepare_few_shot(self, ds_wrapper: Any) -> tuple:
        """Prepare few-shot examples for the classification task."""
        def get_sample_for_class(cl):
            samples = ds_wrapper.dataset_training.filter(
                lambda r: r[ds_wrapper.dataset_info.answer] == cl
            )
            return [samples[random.randint(0, len(samples) - 1)]]

        classes = list(set(ds_wrapper.dataset_training[ds_wrapper.dataset_info.answer]))
        selected_sample = [get_sample_for_class(cl) for cl in classes]

        return (
            self._format_fewshot(selected_sample, ds_wrapper.prompt["prompt"],
                                 ds_wrapper.prompt["answer_format"]),
            self._format_fewshot(selected_sample, ds_wrapper.calibration_prompt["prompt"],
                                 ds_wrapper.prompt["answer_format"])
        )

    @staticmethod
    def _format_fewshot(samples: List[Any],
                        query_format: str, answer_format: str) -> List[Dict[str, str]]:
        """Format few-shot examples."""
        formatted_samples = []
        for sample in samples:
            formatted_samples.extend([
                {"role": "user", "content": query_format.format(sample['query'])},
                {"role": "assistant", "content": answer_format.format(sample['answer'])}
            ])
        return formatted_samples

    def _create_prompts_and_calib_prompts(
        self, batch: Any, ds_wrapper: Any, few_shot_data: tuple
    ) -> tuple:
        """Create prompts and calibration prompts."""
        prompts = self._create_prompts(
            batch[ds_wrapper.dataset_info.query],
            ds_wrapper.prompt, few_shot_data[0]
        )

        calib_prompts = self._create_prompts(
            batch[ds_wrapper.dataset_info.query],
            ds_wrapper.calibration_prompt, few_shot_data[1]
        )
        return prompts, calib_prompts

    def _create_prompts(self, queries: List[Any], prompt_config: Dict[str, str],
                        few_shot: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        """Create prompts from query and prompt configuration."""
        return [
            [
                {"role": "system", "content": prompt_config["system_prompt"]},
                *few_shot,
                {"role": "user", "content": prompt_config["prompt"].format(c)},
            ]
            for c in queries
        ]

    def _compute_option_logprobs(self, calib_prompts: List[List[Dict[str, str]]],
                                 num_choice: int, ds_wrapper: Any) -> List[float]:
        """Compute log probabilities for each option."""
        option_logprobs, _ = self.config.pipeline_config.infer_pipeline.compute_logprob_and_length(
            calib_prompts * num_choice,
            [ds_wrapper.dataset_info.label[choice] for choice in range(num_choice)
             for _ in range(len(calib_prompts))],
        )
        return option_logprobs

    @staticmethod
    def _process_option_probs(option_logprobs: List[float], num_choice: int,
                              num_prompts: int) -> List[List[float]]:
        """Process option probabilities."""
        return [
            [option_logprobs[i + opt * num_prompts] for opt in range(num_choice)]
            for i in range(num_prompts)
        ]

    def _update_classification_data(self, params: ClassificationDataUpdateParams) -> None:
        """Update ClassificationData with batch results."""
        params.data.update(
            predictions=params.results,
            references=[x.item() for x in params.batch[params.ds_wrapper.dataset_info.answer]],
            generation_probs=params.batch_info.logprobs,
            option_probs=self._process_option_probs(
                params.batch_info.option_logprobs, params.num_choice, params.num_prompts
            )
        )

    def _save_intermediate_results(self, saving_fn: Callable, ds_wrapper: Any) -> None:
        """Save intermediate results."""
        saving_fn(self._classification_data, is_final=False, ds_wrapper=ds_wrapper)

    def _save_final_results(self, saving_fn: Callable, ds_wrapper: Any) -> None:
        """Save final results."""
        saving_fn(self._classification_data, is_final=True, ds_wrapper=ds_wrapper)
