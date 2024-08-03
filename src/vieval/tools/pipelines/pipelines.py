import ast
import torch
import os
import json
from tqdm import tqdm
from ..utils.model import get_model
from ..wrapper import OpenAIWrapper, TGIWrapper, GeminiWrapper, HFWrapper
from ..utils.utils import *
from .metric_pipelines import MetricPipeline


class EvalPipeline:
    def __init__(self, task, config):

        # Load generation configuration
        with open(
            os.path.join(config.config_dir, config.lang, "generation_config.json"), "r"
        ) as f:
            GenerationConfig = json.load(f)

        with open(
            os.path.join(config.config_dir, config.lang, "llm_template.json"), "r"
        ) as f:
            LLM_TEMPLATE = json.load(f)

        with open(
            os.path.join(config.config_dir, config.lang, "metric_configuration.json"),
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
            return self.__question_answering(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "summarization":
            return self.__summarization(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif "translation" in task:
            return self.__translation(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif "language-modeling" in task:
            return self.__language_modeling(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif "text-classification" in task:
            return self.__multiple_choice_text_classification(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "sentiment-analysis":
            return self.__multiple_choice_sentiment(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "toxicity-detection":
            return self.__multiple_choice_toxicity(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "knowledge-mtpchoice":
            return self.__multiple_choice(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif task == "knowledge-openended":
            return self.__question_answering_without_context(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "information-retrieval":
            return self.__information_retrieval(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "reasoning":
            return self.__reasoning(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif task == "math":
            return self.__math(ds_wrapper, ds_loader, saving_fn, start_idx)
        else:
            raise NotImplementedError

    def __question_answering(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        original_few_shot = []
        selected_sample = []
        if self.continue_infer_data is not None:
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
        idx = 0
        if self.few_shot:

            def preprocessing_a_record(rec):
                return [
                    rec[ds_wrapper.dataset_info.context],
                    rec[ds_wrapper.dataset_info.query],
                    rec[ds_wrapper.dataset_info.answer]["text"][0],
                ]

            selected_sample_idx = list(
                random.sample(
                    range(len(ds_wrapper.dataset_training)), self.config.num_fs
                )
            )
            selected_sample = [
                preprocessing_a_record(ds_wrapper.dataset_training[s])
                for s in selected_sample_idx
            ]

            original_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.prompt["prompt"],
                answer_format=ds_wrapper.prompt["answer_format"],
            )
        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                [
                    {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
                    *original_few_shot,
                    {
                        "role": "user",
                        "content": ds_wrapper.prompt["prompt"].format(
                            c,
                            q,
                        ),
                    },
                ]
                for c, q in zip(
                    batch[ds_wrapper.dataset_info.context],
                    batch[ds_wrapper.dataset_info.query],
                )
            ]

            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
            predictions.extend(results)
            references.extend(
                [x[0] for x in batch[ds_wrapper.dataset_info.answer]["text"]]
            )
            generation_probs.extend(logprobs)

            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {
                    "predictions": predictions,
                    "references": references,
                    "generation_probs": generation_probs,
                    "fewshot": selected_sample,
                }
                saving_fn(generations)
                mean_result = self.metric_pipeline.run_mean(
                    generations,
                    self.task_name,
                    ds_wrapper.prompt["answer_key"],
                    ds_wrapper.dataset_info.label,
                    self.config,
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "fewshot": selected_sample,
        }
        mean_result = self.metric_pipeline.run_mean(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        std_result = self.metric_pipeline.run_std(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __question_answering_without_context(
        self, ds_wrapper, ds_loader, saving_fn, start_idx=0
    ):
        predictions = []
        references = []
        generation_probs = []
        calib_probs = []
        idx = 0
        original_few_shot = []
        calib_few_shot = []
        selected_sample = []
        if self.continue_infer_data is not None:
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
            calib_probs.extend(self.continue_infer_data["calibration_probs"])
        if self.few_shot:

            def preprocessing_a_record(rec):
                return [
                    rec[ds_wrapper.dataset_info.query],
                    rec[ds_wrapper.dataset_info.answer],
                ]

            selected_sample_idx = list(
                random.sample(
                    range(len(ds_wrapper.dataset_training)), self.config.num_fs
                )
            )
            selected_sample = [
                preprocessing_a_record(ds_wrapper.dataset_training[s])
                for s in selected_sample_idx
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

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                [
                    {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
                    *original_few_shot,
                    {
                        "role": "user",
                        "content": ds_wrapper.prompt["prompt"].format(
                            q,
                        ),
                    },
                ]
                for q in batch[ds_wrapper.dataset_info.query]
            ]

            calib_prompts = [
                [
                    {
                        "role": "system",
                        "content": ds_wrapper.calibration_prompt["system_prompt"],
                    },
                    *calib_few_shot,
                    {
                        "role": "user",
                        "content": ds_wrapper.calibration_prompt["prompt"].format(
                            q,
                        ),
                    },
                ]
                for q in batch[ds_wrapper.dataset_info.query]
            ]

            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
            calibprob_batch, _ = self.infer_pipeline.compute_logprob_and_length(
                calib_prompts, batch[ds_wrapper.dataset_info.answer]
            )
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.dataset_info.answer]])
            generation_probs.extend(logprobs)
            calib_probs.extend(calibprob_batch)
            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {
                    "predictions": predictions,
                    "references": references,
                    "generation_probs": generation_probs,
                    "calibration_probs": calib_probs,
                    "fewshot": selected_sample,
                }

                saving_fn(generations)
                mean_result = self.metric_pipeline.run_mean(
                    generations,
                    self.task_name,
                    ds_wrapper.prompt["answer_key"],
                    ds_wrapper.dataset_info.label,
                    self.config,
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "calibration_probs": calib_probs,
            "fewshot": selected_sample,
        }
        mean_result = self.metric_pipeline.run_mean(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        std_result = self.metric_pipeline.run_std(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __summarization(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        original_documents = []
        predictions = []
        original_few_shot = []
        selected_sample = []
        references = []
        generation_probs = []
        if self.continue_infer_data is not None:
            original_documents.extend(self.continue_infer_data["original_documents"])
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
        idx = 0
        if self.few_shot:

            def preprocessing_a_record(rec):
                return [
                    rec[ds_wrapper.dataset_info.source],
                    rec[ds_wrapper.dataset_info.target],
                ]

            selected_sample_idx = list(
                random.sample(
                    range(len(ds_wrapper.dataset_training)), self.config.num_fs
                )
            )
            selected_sample = [
                preprocessing_a_record(ds_wrapper.dataset_training[s])
                for s in selected_sample_idx
            ]

            original_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.prompt["prompt"],
                answer_format=ds_wrapper.prompt["answer_format"],
            )
        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                [
                    {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
                    *original_few_shot,
                    {
                        "role": "user",
                        "content": ds_wrapper.prompt["prompt"].format(
                            document,
                        ),
                    },
                ]
                for document in batch[ds_wrapper.dataset_info.source]
            ]
            original_documents.extend(
                [x for x in batch[ds_wrapper.dataset_info.source]]
            )

            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.dataset_info.target]])
            generation_probs.extend(logprobs)

            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {
                    "original_documents": original_documents,
                    "predictions": predictions,
                    "references": references,
                    "generation_probs": generation_probs,
                    "fewshot": selected_sample,
                }
                saving_fn(generations)
                mean_result = self.metric_pipeline.run_mean(
                    generations,
                    self.task_name,
                    ds_wrapper.prompt["answer_key"],
                    ds_wrapper.dataset_info.label,
                    self.config,
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {
            "original_documents": original_documents,
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "fewshot": selected_sample,
        }
        mean_result = self.metric_pipeline.run_mean(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        std_result = self.metric_pipeline.run_std(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __multiple_choice_sentiment(
        self, ds_wrapper, ds_loader, saving_fn, start_idx=0
    ):
        predictions = []
        references = []
        generation_probs = []
        option_probs = []
        idx = 0
        original_few_shot = []
        calib_few_shot = []
        selected_sample = []
        num_choice = len(ds_wrapper.dataset_info.label)
        if self.continue_infer_data is not None:
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
            option_probs.extend(self.continue_infer_data["option_probs"])
        if self.few_shot:

            def preprocessing_a_record(rec):
                return [
                    rec[ds_wrapper.dataset_info.query],
                    rec[ds_wrapper.dataset_info.answer],
                ]

            classes = unique(
                ds_wrapper.dataset_training[ds_wrapper.dataset_info.answer]
            )
            selected_sample = []
            for cl in classes:
                cl_samples = ds_wrapper.dataset_training.filter(
                    lambda r: r[ds_wrapper.dataset_info.answer] == cl
                )
                selected_sample.append(
                    preprocessing_a_record(
                        cl_samples[random.randint(0, len(cl_samples))]
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

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                [
                    {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
                    *original_few_shot,
                    {
                        "role": "user",
                        "content": ds_wrapper.prompt["prompt"].format(
                            c,
                        ),
                    },
                ]
                for c in batch[ds_wrapper.dataset_info.query]
            ]
            calib_prompts = [
                [
                    {
                        "role": "system",
                        "content": ds_wrapper.calibration_prompt["system_prompt"],
                    },
                    *calib_few_shot,
                    {
                        "role": "user",
                        "content": ds_wrapper.calibration_prompt["prompt"].format(
                            c,
                        ),
                    },
                ]
                for c in batch[ds_wrapper.dataset_info.query]
            ]
            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)

            option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                calib_prompts * num_choice,
                [
                    ds_wrapper.dataset_info.label[choice]
                    for choice in range(num_choice)
                    for _ in range(len(prompts))
                ],
            )
            predictions.extend(results)
            references.extend([x.item() for x in batch[ds_wrapper.dataset_info.answer]])
            generation_probs.extend(logprobs)
            option_probs.extend(
                [
                    [
                        option_logprobs[i + opt * len(prompts)]
                        for opt in range(num_choice)
                    ]
                    for i in range(len(prompts))
                ]
            )
            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {
                    "predictions": predictions,
                    "references": references,
                    "generation_probs": generation_probs,
                    "option_probs": option_probs,
                    "fewshot": selected_sample,
                }
                saving_fn(generations)
                mean_result = self.metric_pipeline.run_mean(
                    generations,
                    self.task_name,
                    ds_wrapper.prompt["answer_key"],
                    ds_wrapper.dataset_info.label,
                    self.config,
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "option_probs": option_probs,
            "fewshot": selected_sample,
        }

        mean_result = self.metric_pipeline.run_mean(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        std_result = self.metric_pipeline.run_std(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __multiple_choice_text_classification(
        self, ds_wrapper, ds_loader, saving_fn, start_idx=0
    ):
        predictions = []
        references = []
        generation_probs = []
        option_probs = []
        if self.continue_infer_data is not None:
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
            option_probs.extend(self.continue_infer_data["option_probs"])
        idx = 0
        original_few_shot = []
        calib_few_shot = []
        selected_sample = []
        num_choice = len(ds_wrapper.dataset_info.label)

        if self.few_shot:

            def preprocessing_a_record(rec):
                return [
                    rec[ds_wrapper.dataset_info.query],
                    rec[ds_wrapper.dataset_info.answer],
                ]

            classes = unique(
                ds_wrapper.dataset_training[ds_wrapper.dataset_info.answer]
            )

            selected_sample = []
            for cl in classes:
                cl_samples = ds_wrapper.dataset_training.filter(
                    lambda r: (r[ds_wrapper.dataset_info.answer] == cl)
                )
                selected_sample.append(
                    cl_samples[random.randint(0, len(cl_samples) - 1)]
                )

            selected_sample = [preprocessing_a_record(x) for x in selected_sample]
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

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                [
                    {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
                    *original_few_shot,
                    {
                        "role": "user",
                        "content": ds_wrapper.prompt["prompt"].format(
                            c,
                        ),
                    },
                ]
                for c in batch[ds_wrapper.dataset_info.query]
            ]

            calib_prompts = [
                [
                    {
                        "role": "system",
                        "content": ds_wrapper.calibration_prompt["system_prompt"],
                    },
                    *calib_few_shot,
                    {
                        "role": "user",
                        "content": ds_wrapper.calibration_prompt["prompt"].format(
                            c,
                        ),
                    },
                ]
                for c in batch[ds_wrapper.dataset_info.query]
            ]

            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)

            option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                calib_prompts * num_choice,
                [
                    ds_wrapper.dataset_info.label[choice]
                    for choice in range(num_choice)
                    for _ in range(len(prompts))
                ],
            )
            predictions.extend(results)
            references.extend(
                [
                    eval(x) if type(x) is str else x.item()
                    for x in batch[ds_wrapper.dataset_info.answer]
                ]
            )
            generation_probs.extend(logprobs)
            option_probs.extend(
                [
                    [
                        option_logprobs[i + opt * len(prompts)]
                        for opt in range(num_choice)
                    ]
                    for i in range(len(prompts))
                ]
            )
            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {
                    "predictions": predictions,
                    "references": references,
                    "generation_probs": generation_probs,
                    "option_probs": option_probs,
                    "fewshot": selected_sample,
                }
                saving_fn(generations)
                mean_result = self.metric_pipeline.run_mean(
                    generations,
                    self.task_name,
                    ds_wrapper.prompt["answer_key"],
                    ds_wrapper.dataset_info.label,
                    self.config,
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "option_probs": option_probs,
            "fewshot": selected_sample,
        }
        mean_result = self.metric_pipeline.run_mean(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        std_result = self.metric_pipeline.run_std(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __multiple_choice_toxicity(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        option_probs = []
        idx = 0
        original_few_shot = []
        calib_few_shot = []
        selected_sample = []
        num_choice = len(ds_wrapper.dataset_info.label)
        if self.continue_infer_data is not None:
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
            option_probs.extend(self.continue_infer_data["option_probs"])
        if self.few_shot:

            def preprocessing_a_record(rec):
                return [
                    rec[ds_wrapper.dataset_info.query],
                    rec[ds_wrapper.dataset_info.answer],
                ]

            classes = unique(
                ds_wrapper.dataset_training[ds_wrapper.dataset_info.answer]
            )
            selected_sample = []
            for cl in classes:
                cl_samples = ds_wrapper.dataset_training.filter(
                    lambda r: r[ds_wrapper.dataset_info.answer] == cl
                )
                selected_sample.append(
                    preprocessing_a_record(
                        cl_samples[random.randint(0, len(cl_samples))]
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

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                [
                    {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
                    *original_few_shot,
                    {
                        "role": "user",
                        "content": ds_wrapper.prompt["prompt"].format(
                            c,
                        ),
                    },
                ]
                for c in batch[ds_wrapper.dataset_info.query]
            ]

            calib_prompts = [
                [
                    {
                        "role": "system",
                        "content": ds_wrapper.calibration_prompt["system_prompt"],
                    },
                    *calib_few_shot,
                    {
                        "role": "user",
                        "content": ds_wrapper.calibration_prompt["prompt"].format(
                            c,
                        ),
                    },
                ]
                for c in batch[ds_wrapper.dataset_info.query]
            ]
            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)

            option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                calib_prompts * num_choice,
                [
                    ds_wrapper.dataset_info.label[choice]
                    for choice in range(num_choice)
                    for _ in range(len(prompts))
                ],
            )
            predictions.extend(results)
            references.extend([x.item() for x in batch[ds_wrapper.dataset_info.answer]])
            generation_probs.extend(logprobs)
            option_probs.extend(
                [
                    [
                        option_logprobs[i + opt * len(prompts)]
                        for opt in range(num_choice)
                    ]
                    for i in range(len(prompts))
                ]
            )
            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {
                    "predictions": predictions,
                    "references": references,
                    "generation_probs": generation_probs,
                    "option_probs": option_probs,
                    "fewshot": selected_sample,
                }
                saving_fn(generations)
                mean_result = self.metric_pipeline.run_mean(
                    generations,
                    self.task_name,
                    ds_wrapper.prompt["answer_key"],
                    ds_wrapper.dataset_info.label,
                    self.config,
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "option_probs": option_probs,
            "fewshot": selected_sample,
        }
        mean_result = self.metric_pipeline.run_mean(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        std_result = self.metric_pipeline.run_std(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __multiple_choice(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        def format_list_ans(ans_list):
            return "\n".join(
                list(
                    map(
                        lambda ans: f"{ds_wrapper.dataset_info.label[ans[0]]}: ''' {ans[1]} '''",
                        enumerate(ans_list),
                    )
                )
            )

        predictions = []
        references = []
        generation_probs = []
        option_probs = []
        idx = 0
        original_few_shot = []
        calib_few_shot = []
        option_order_all = []
        selected_sample = []
        # alphabet2idx = {chr(i + 65): i for i in range(26)}
        num_choice = len(ds_wrapper.dataset_info.label)
        if self.continue_infer_data is not None:
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
            option_probs.extend(self.continue_infer_data["option_probs"])
            option_order_all.extend(self.continue_infer_data["option_orders"])

        if self.few_shot:

            def preprocessing_a_record(rec):
                return [
                    rec[ds_wrapper.dataset_info.context],
                    rec[ds_wrapper.dataset_info.query],
                    format_list_ans(
                        ast.literal_eval(rec[ds_wrapper.dataset_info.options])
                    ),
                    rec[ds_wrapper.dataset_info.answer],
                ]

            selected_sample_idx = list(
                random.sample(range(len(ds_wrapper.dataset_training)), 2)
            )
            selected_sample = [
                preprocessing_a_record(ds_wrapper.dataset_training[s])
                for s in selected_sample_idx
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

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = []
            calib_prompts = []
            remap_order_batch = []
            for cq in zip(
                batch[ds_wrapper.dataset_info.context],
                batch[ds_wrapper.dataset_info.query],
                batch[ds_wrapper.dataset_info.options],
            ):

                c = cq[0]
                q = cq[1]
                opts = ast.literal_eval(cq[2])
                order_shuffle = list(range(len(opts)))
                if ds_wrapper.dataset_info.random:
                    random.shuffle(order_shuffle)
                remap_order_batch.append(order_shuffle)
                new_opts = [opts[i] for i in order_shuffle]
                prompts.append(
                    [
                        {
                            "role": "system",
                            "content": ds_wrapper.prompt["system_prompt"],
                        },
                        *original_few_shot,
                        {
                            "role": "user",
                            "content": ds_wrapper.prompt["prompt"].format(
                                c,
                                q,
                                format_list_ans(new_opts),
                            ),
                        },
                    ]
                )
                calib_prompts.append(
                    [
                        {
                            "role": "system",
                            "content": ds_wrapper.calibration_prompt["system_prompt"],
                        },
                        *calib_few_shot,
                        {
                            "role": "user",
                            "content": ds_wrapper.calibration_prompt["prompt"].format(
                                c,
                                q,
                                format_list_ans(new_opts),
                            ),
                        },
                    ]
                )

            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
            option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                calib_prompts * num_choice,
                [
                    ds_wrapper.dataset_info.label[choice]
                    for choice in range(num_choice)
                    for _ in range(len(prompts))
                ],
            )
            opt_calib_out = [
                [option_logprobs[i + opt * len(prompts)] for opt in range(num_choice)]
                for i in range(len(prompts))
            ]

            # REsort answer of calib
            # opt_calib_out = [k for k, _ in sorted(zip(opt_calib_out, remap_order), key=lambda x: x[1])]
            option_order_all.extend(remap_order_batch)
            predictions.extend(results)
            # In case order of options is changed
            # Map the reference to the new order
            references.extend(
                [
                    ds_wrapper.dataset_info.label[
                        remap.index(ds_wrapper.dataset_info.label.index(x))
                    ]
                    for x, remap in zip(
                        batch[ds_wrapper.dataset_info.answer], remap_order_batch
                    )
                ]
            )

            generation_probs.extend(logprobs)
            option_probs.extend(opt_calib_out)
            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {
                    "predictions": predictions,
                    "references": references,  # new order
                    "generation_probs": generation_probs,
                    "option_probs": option_probs,  # new order
                    "option_orders": option_order_all,
                    "fewshot": selected_sample,
                }
                saving_fn(generations)
                mean_result = self.metric_pipeline.run_mean(
                    generations,
                    self.task_name,
                    ds_wrapper.prompt["answer_key"],
                    ds_wrapper.dataset_info.label,
                    self.config,
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "option_probs": option_probs,
            "option_orders": option_order_all,
            "fewshot": selected_sample,
        }

        mean_result = self.metric_pipeline.run_mean(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        std_result = self.metric_pipeline.run_std(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __language_modeling(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        if self.continue_infer_data is not None:
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
        idx = 0
        original_few_shot = []
        selected_sample = []
        if self.few_shot:

            def preprocessing_a_record(rec):
                return [
                    rec[ds_wrapper.dataset_info.source],
                    rec[ds_wrapper.dataset_info.target],
                ]

            selected_sample = [
                preprocessing_a_record(s) for s in ds_wrapper.dataset_training
            ]
            original_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.prompt["prompt"],
                answer_format=ds_wrapper.prompt["answer_format"],
            )

        # Create few-shot strings
        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                [
                    {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
                    *original_few_shot,
                    {
                        "role": "user",
                        "content": ds_wrapper.prompt["prompt"].format(
                            c,
                        ),
                    },
                ]
                for c in batch[ds_wrapper.dataset_info.source]
            ]

            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.dataset_info.target]])
            generation_probs.extend(logprobs)

            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {
                    "predictions": predictions,
                    "references": references,
                    "generation_probs": generation_probs,
                    "fewshot": selected_sample,
                }
                saving_fn(generations)
                mean_result = self.metric_pipeline.run_mean(
                    generations,
                    self.task_name,
                    ds_wrapper.prompt["answer_key"],
                    ds_wrapper.dataset_info.label,
                    self.config,
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "fewshot": selected_sample,
        }
        mean_result = self.metric_pipeline.run_mean(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        std_result = self.metric_pipeline.run_std(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __information_retrieval(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        # sub_task = self.task.split("_")[1]
        idx = 0
        original_few_shot = []
        calib_few_shot = []
        selected_sample = []
        if self.few_shot:

            def preprocessing_a_record(rec):
                return [
                    rec[ds_wrapper.dataset_info.passages],
                    rec[ds_wrapper.dataset_info.query],
                    rec[ds_wrapper.dataset_info.answer],
                ]

            random_sample = list(random.sample(list(ds_wrapper.dataset_training), 1))[0]
            # random_batch_passages = random_sample[ds_wrapper.dataset_info.passages]
            # if sub_task == "mmarco":
            #     ref_passage_id = random_sample[ds_wrapper.dataset_info.answer][0]
            #     ref_passage_idx = random_batch_passages["id"].index(
            #         ref_passage_id)
            #     rnd_passage_idx = random.choice(
            #         [
            #             i
            #             for i in range(len(random_batch_passages["id"]))
            #             if i != ref_passage_idx
            #         ]
            #     )

            # else:
            #     ref_passage_id = random_sample[ds_wrapper.dataset_info.answer][0]
            #     ref_passage_idx = random_batch_passages["id"].index(
            #         ref_passage_id)
            #     rnd_passage_id = random_sample[ds_wrapper.dataset_info.answer][-1]
            #     rnd_passage_idx = batch_passages["id"].index(rnd_passage_id)

            first_sample = {
                "passages": random_sample["positive"],
                "query": random_sample[ds_wrapper.dataset_info.query],
                "references": "Yes",
            }
            second_sample = {
                "passages": random_sample["negative"],
                "query": random_sample[ds_wrapper.dataset_info.query],
                "references": "No",
            }

            selected_sample = [
                preprocessing_a_record(s) for s in [first_sample, second_sample]
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

        BATCH_PASSAGE_SIZE = 10
        # Create few-shot strings
        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue
            for query_with_a_batch_passages in range(len(batch[ds_wrapper.id])):
                query_id = batch[ds_wrapper.id][query_with_a_batch_passages]
                query = batch[ds_wrapper.dataset_info.query][
                    query_with_a_batch_passages
                ]
                try:
                    ref_passage_id = batch[ds_wrapper.dataset_info.answer][0][
                        query_with_a_batch_passages
                    ]
                except:
                    if len(list(batch[ds_wrapper.dataset_info.answer])) < 1:
                        continue
                    ref_passage_id = list(batch[ds_wrapper.dataset_info.answer][0])[
                        query_with_a_batch_passages
                    ]
                batch_passages = batch[ds_wrapper.dataset_info.passages]

                top30_passage_ids = column(
                    batch_passages["id"], query_with_a_batch_passages
                )
                top30_passages = column(
                    batch_passages["passage"], query_with_a_batch_passages
                )
                for psg in range(0, len(top30_passage_ids), BATCH_PASSAGE_SIZE):
                    prompts = [
                        [
                            {
                                "role": "system",
                                "content": ds_wrapper.prompt["system_prompt"],
                            },
                            *original_few_shot,
                            {
                                "role": "user",
                                "content": ds_wrapper.prompt["prompt"].format(
                                    p,
                                    query,
                                ),
                            },
                        ]
                        for p in top30_passages[psg : psg + BATCH_PASSAGE_SIZE]
                    ]
                    calib_prompts = [
                        [
                            {
                                "role": "system",
                                "content": ds_wrapper.calibration_prompt[
                                    "system_prompt"
                                ],
                            },
                            *calib_few_shot,
                            {
                                "role": "user",
                                "content": ds_wrapper.calibration_prompt[
                                    "prompt"
                                ].format(
                                    p,
                                    query,
                                ),
                            },
                        ]
                        for p in top30_passages[psg : psg + BATCH_PASSAGE_SIZE]
                    ]
                    results, logprobs, _ = self.infer_pipeline(
                        prompts, return_probs=True
                    )

                    option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                        calib_prompts * 2,
                        [
                            choice
                            for choice in ["Yes", "No"]
                            for _ in range(len(prompts))
                        ],
                    )
                    save_each_prompt = list(
                        map(
                            lambda x, y, z, t, q: {
                                "query_id": (
                                    query_id.item()
                                    if type(query_id) is not str
                                    else query_id
                                ),
                                "query": query,
                                "passage_id": z.item() if type(z) is not str else z,
                                "passage": t,
                                "label": int(
                                    z.item() == ref_passage_id
                                    if type(z) is not str
                                    else z == ref_passage_id
                                ),
                                "prediction": x,
                                "generation_probs": y,
                                "calib_probs": [
                                    option_logprobs[q + opt * len(prompts)]
                                    for opt in range(2)
                                ],
                            },
                            results,
                            logprobs,
                            top30_passage_ids[psg : psg + BATCH_PASSAGE_SIZE],
                            top30_passages[psg : psg + BATCH_PASSAGE_SIZE],
                            range(len(prompts)),
                        )
                    )
                    predictions.extend(save_each_prompt)

            idx += 1

            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {"fewshot": selected_sample, "predictions": predictions}
                saving_fn(generations)
                mean_result = self.metric_pipeline.run_mean(
                    generations,
                    self.task_name,
                    ds_wrapper.prompt["answer_key"],
                    ds_wrapper.dataset_info.label,
                    self.config,
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {"fewshot": selected_sample, "predictions": predictions}
        mean_result = self.metric_pipeline.run_mean(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        std_result = self.metric_pipeline.run_std(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __reasoning(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        calib_probs = []
        idx = 0
        original_few_shot = []
        calib_few_shot = []
        selected_sample = []

        if self.continue_infer_data is not None:
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
            calib_probs.extend(self.continue_infer_data["calibration_probs"])

        if self.few_shot:

            def preprocessing_a_record(rec):
                return [
                    rec[ds_wrapper.dataset_info.query],
                    rec[ds_wrapper.dataset_info.answer],
                ]

            selected_sample = [
                preprocessing_a_record(s)
                for s in list(
                    random.sample(list(ds_wrapper.dataset_training), self.config.num_fs)
                )
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
        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                [
                    {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
                    *original_few_shot,
                    {
                        "role": "user",
                        "content": ds_wrapper.prompt["prompt"].format(rule),
                    },
                ]
                for rule in batch[ds_wrapper.dataset_info.query]
            ]
            calib_prompts = [
                [
                    {
                        "role": "system",
                        "content": ds_wrapper.calibration_prompt["system_prompt"],
                    },
                    *calib_few_shot,
                    {
                        "role": "user",
                        "content": ds_wrapper.calibration_prompt["prompt"].format(rule),
                    },
                ]
                for rule in batch[ds_wrapper.dataset_info.query]
            ]

            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
            calibprob_batch, _ = self.infer_pipeline.compute_logprob_and_length(
                calib_prompts, batch[ds_wrapper.dataset_info.answer]
            )
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.dataset_info.answer]])
            generation_probs.extend(logprobs)
            calib_probs.extend(calibprob_batch)

            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {
                    "predictions": predictions,
                    "references": references,
                    "generation_probs": generation_probs,
                    "calibration_probs": calib_probs,
                    "fewshot": selected_sample,
                }

                saving_fn(generations)
                mean_result = self.metric_pipeline.run_mean(
                    generations,
                    self.task_name,
                    ds_wrapper.prompt["answer_key"],
                    ds_wrapper.dataset_info.label,
                    self.config,
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "calibration_probs": calib_probs,
            "fewshot": selected_sample,
        }

        mean_result = self.metric_pipeline.run_mean(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        std_result = self.metric_pipeline.run_std(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )

        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __math(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        calib_probs = []
        math_problem_type = []
        idx = 0
        original_few_shot = []
        calib_few_shot = []
        selected_sample = []
        pattern = regex.compile(r"\\boxed\{(?:[^{}]|(?R))*\}")
        # res_list = pattern.findall(text)
        # return res_list[0] if res_list else None
        if self.continue_infer_data is not None:
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
            calib_probs.extend(self.continue_infer_data["calibration_probs"])
            mat_problem_type.extend(
                self.continue_infer_data.get("math_problem_type", [])
            )
        if self.few_shot:

            def preprocessing_a_record(rec):
                return [
                    rf"{rec[ds_wrapper.dataset_info.query]}",
                    rf"{rec[ds_wrapper.dataset_info.answer]}",
                ]

            selected_sample = [
                preprocessing_a_record(s)
                for s in list(
                    random.sample(list(ds_wrapper.dataset_training), self.config.num_fs)
                )
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

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue
            prompts = [
                [
                    {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
                    *original_few_shot,
                    {
                        "role": "user",
                        "content": ds_wrapper.prompt["prompt"].format(rf"{rule}"),
                    },
                ]
                for rule in batch[ds_wrapper.dataset_info.query]
            ]
            calib_prompts = [
                [
                    {
                        "role": "system",
                        "content": ds_wrapper.calibration_prompt["system_prompt"],
                    },
                    *calib_few_shot,
                    {
                        "role": "user",
                        "content": ds_wrapper.calibration_prompt["prompt"].format(
                            rf"{rule}"
                        ),
                    },
                ]
                for rule in batch[ds_wrapper.dataset_info.query]
            ]

            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
            calibprob_batch, _ = self.infer_pipeline.compute_logprob_and_length(
                calib_prompts, batch[ds_wrapper.dataset_info.answer]
            )
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.dataset_info.answer]])
            generation_probs.extend(logprobs)
            calib_probs.extend(calibprob_batch)
            math_problem_type.extend(
                [x for x in batch[ds_wrapper.dataset_info.type_id]]
            )
            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {
                    "predictions": predictions,
                    "references": references,
                    "generation_probs": generation_probs,
                    "calibration_probs": calib_probs,
                    "fewshot": selected_sample,
                    "math_problem_type": math_problem_type,
                }

                saving_fn(generations)
                mean_result = self.metric_pipeline.run_mean(
                    generations,
                    self.task_name,
                    ds_wrapper.prompt["answer_key"],
                    ds_wrapper.dataset_info.label,
                    self.config,
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "calibration_probs": calib_probs,
            "fewshot": selected_sample,
            "math_problem_type": math_problem_type,
        }

        mean_result = self.metric_pipeline.run_mean(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        std_result = self.metric_pipeline.run_std(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )

        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __translation(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        idx = 0
        original_few_shot = []
        selected_sample = []
        if self.continue_infer_data is not None:
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
        if self.few_shot:

            def preprocessing_a_record(rec):
                return [
                    rec[ds_wrapper.dataset_info.source],
                    rec[ds_wrapper.dataset_info.target],
                ]

            selected_sample = [
                preprocessing_a_record(s)
                for s in list(
                    random.sample(list(ds_wrapper.dataset_training), self.config.num_fs)
                )
            ]
            original_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.prompt["prompt"],
                answer_format=ds_wrapper.prompt["answer_format"],
            )

        # Create few-shot strings
        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                [
                    {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
                    *original_few_shot,
                    {
                        "role": "user",
                        "content": ds_wrapper.prompt["prompt"].format(
                            document,
                        ),
                    },
                ]
                for document in batch[ds_wrapper.dataset_info.source]
            ]

            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.dataset_info.target]])
            generation_probs.extend(logprobs)

            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {
                    "predictions": predictions,
                    "references": references,
                    "generation_probs": generation_probs,
                    "fewshot": selected_sample,
                }
                saving_fn(generations)
                mean_result = self.metric_pipeline.run_mean(
                    generations,
                    self.task_name,
                    ds_wrapper.prompt["answer_key"],
                    ds_wrapper.dataset_info.label,
                    self.config,
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "fewshot": selected_sample,
        }
        mean_result = self.metric_pipeline.run_mean(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        std_result = self.metric_pipeline.run_std(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

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
