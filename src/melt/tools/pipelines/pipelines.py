"pipelines"
import os
import json
import torch
from melt.tools.wrapper import (
    OpenAIWrapper,
    TGIWrapper,
    GeminiWrapper,
    VLLMWrapper,
    HFWrapper,
)
from melt.tools.pipelines.metric_pipelines import MetricPipeline
from melt.tools.pipelines.__question_answering import __question_answering
from melt.tools.pipelines.__question_answering_without_context import (
    __question_answering_without_context
)
from melt.tools.pipelines.__summarization import __summarization
from melt.tools.pipelines.__multiple_choice_sentiment import __multiple_choice_sentiment
from melt.tools.pipelines.__multiple_choice_text_classification import (
    __multiple_choice_text_classification)
from melt.tools.pipelines.__multiple_choice_toxicity import __multiple_choice_toxicity
from melt.tools.pipelines.__multiple_choice import __multiple_choice
from melt.tools.pipelines.__language_modeling import __language_modeling
from melt.tools.pipelines.__information_retrieval import __information_retrieval
from melt.tools.pipelines.__reasoning import __reasoning
from melt.tools.pipelines.__math import __math
from melt.tools.pipelines.__translation import __translation
class EvalPipeline:
    "class"
    def __init__(self, task, config):
        # Load generation configuration
        with open(
            os.path.join(
                config.config_dir, config.lang, "generation_config.json"), "r", encoding="utf-8"
        ) as f:
            generation_config = json.load(f)

        with open(
            os.path.join(config.config_dir, "llm_template.json"), "r", encoding="utf-8"
        ) as f:
            llm_template = json.load(f)

        with open(
            os.path.join(
                config.config_dir, config.lang, "metric_configuration.json"), "r", encoding="utf-8"
        ) as f:
            metric_config = json.load(f)

        # Load task
        self.task_name = task

        # Load pipelines
        if config.wtype == "tgi":
            self.infer_pipeline = TGIWrapper(
                generation_config=generation_config[self.task_name],
                template=llm_template[config.ptemplate],
            )
        elif config.wtype == "hf":
            self.infer_pipeline = HFWrapper(
                config=config,
                generation_config=generation_config[self.task_name],
                template=llm_template[config.ptemplate],
            )
        elif config.wtype == "vllm":
            self.infer_pipeline = VLLMWrapper(
                config=config,
                generation_config=generation_config[self.task_name],
                template=llm_template[config.ptemplate],
            )
        elif config.wtype == "openai":
            self.infer_pipeline = OpenAIWrapper(
                engine=config.model_name,
                generation_config=generation_config[self.task_name],
            )
        elif config.wtype == "gemini":
            self.infer_pipeline = GeminiWrapper(
                model_name=config.model_name,
                generation_config=generation_config[self.task_name],
            )
        else:
            raise ValueError("Invalid wrapper type")

        self.config = config
        self.config.task = self.task_name
        self.config.metric_config = metric_config
        self.few_shot = False
        self.continue_infer_data = None
        self.metric_pipeline = MetricPipeline()
        self.config.filepath = None
        self.generation_results_file = None  # Initialize in __init__

    def __call__(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        task_mapping = {
            "question-answering": __question_answering,
            "summarization": __summarization,
            "translation": __translation,
            "language-modeling": __language_modeling,
            "text-classification": __multiple_choice_text_classification,
            "sentiment-analysis": __multiple_choice_sentiment,
            "toxicity-detection": __multiple_choice_toxicity,
            "knowledge-mtpchoice": __multiple_choice,
            "knowledge-openended": __question_answering_without_context,
            "information-retrieval": __information_retrieval,
            "reasoning": __reasoning,
            "math": __math,
        }

        if self.task_name in task_mapping:
            return task_mapping[self.task_name](ds_wrapper, ds_loader, saving_fn, start_idx)

        raise NotImplementedError  # Removed unnecessary "else"
    def run(
        self,
        ds_wrapper,
        ds_loader,
        generation_results_file,
        saving_fn,
        start_idx=0,
        few_shot=False,
        continue_infer=None,
    ):
        "run"
        self.generation_results_file = generation_results_file
        self.config.filepath = generation_results_file
        self.continue_infer_data = continue_infer
        self.few_shot = few_shot
        with torch.no_grad():
            results = self(ds_wrapper, ds_loader, saving_fn, start_idx)
        return results
