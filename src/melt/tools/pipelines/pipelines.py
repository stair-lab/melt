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
                config.config_dir, config.lang, "generation_config.json"
            ),
            "r",
        ) as f:
            GenerationConfig = json.load(f)

        with open(
            os.path.join(config.config_dir, "llm_template.json"), "r"
        ) as f:
            LLM_TEMPLATE = json.load(f)

        with open(
            os.path.join(
                config.config_dir, config.lang, "metric_configuration.json"
            ),
            "r",
        ) as f:
            METRIC_CONFIG = json.load(f)
        # Load task
        self.task_name = task

        # Load pipelines
        # print(config.tgi)
        if config.wtype == "tgi":
            self.infer_pipeline = TGIWrapper(
                generation_config=GenerationConfig[self.task_name],
                template=LLM_TEMPLATE[config.ptemplate],
            )
        elif config.wtype == "hf":
            self.infer_pipeline = HFWrapper(
                config=config,
                generation_config=GenerationConfig[self.task_name],
                template=LLM_TEMPLATE[config.ptemplate],
            )
        elif config.wtype == "vllm":
            self.infer_pipeline = VLLMWrapper(
                config=config,
                generation_config=GenerationConfig[self.task_name],
                template=LLM_TEMPLATE[config.ptemplate],
            )
        elif config.wtype == "openai":
            self.infer_pipeline = OpenAIWrapper(
                engine=config.model_name,
                generation_config=GenerationConfig[self.task_name],
            )
        elif config.wtype == "gemini":
            self.infer_pipeline = GeminiWrapper(
                model_name=config.model_name,
                generation_config=GenerationConfig[self.task_name],
            )
        else:
            raise ValueError("Invalid wrapper type")

        self.config = config
        self.config.task = self.task_name
        self.config.metric_config = METRIC_CONFIG
        self.few_shot = False
        self.continue_infer_data = None
        # Metric pipeline configuration
        self.metric_pipeline = MetricPipeline()
        self.config.filepath = None
    def __call__(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        task = self.task_name

        if task == "question-answering":
            return __question_answering(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "summarization":
            return __summarization(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif "translation" in task:
            return __translation(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif "language-modeling" in task:
            return __language_modeling(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif "text-classification" in task:
            return __multiple_choice_text_classification(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "sentiment-analysis":
            return __multiple_choice_sentiment(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "toxicity-detection":
            return __multiple_choice_toxicity(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "knowledge-mtpchoice":
            return __multiple_choice(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "knowledge-openended":
            return __question_answering_without_context(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "information-retrieval":
            return __information_retrieval(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "reasoning":
            return __reasoning(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "math":
            return __math(ds_wrapper, ds_loader, saving_fn, start_idx)
        else:
            raise NotImplementedError
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
        self.generation_results_file = generation_results_file
        self.config.filepath = generation_results_file
        self.continue_infer_data = continue_infer
        self.few_shot = few_shot
        with torch.no_grad():
            results = self(ds_wrapper, ds_loader, saving_fn, start_idx)
        return results
