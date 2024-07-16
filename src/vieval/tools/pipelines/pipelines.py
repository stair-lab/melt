import ast
import torch
import os
import json
from tqdm import tqdm
from ..utils.model import get_model
from ..wrapper import AzureGPTWrapper, TGIWrapper, GeminiWrapper, HFWrapper
from ..utils.utils import *
from ..utils.metric_utils import info_from_filename
from .metric_pipelines import MetricPipeline


class EvalPipeline:
    def __init__(self, task, config):

        # Load generation configuration
        with open(os.path.join(config.config_dir, "generation_config.json"), "r") as f:
            GenerationConfig = json.load(f)

        with open(os.path.join(config.config_dir, "llm_template.json"), "r") as f:
            LLM_TEMPLATE = json.load(f)

        # Load task
        self.task = task
        extract_task = self.task.split("_")[0]

        # Load pipelines
        # print(config.tgi)
        if config.wtype == "tgi":
            self.infer_pipeline = TGIWrapper(
                api_endpoint=config.tgi,
                generation_config=GenerationConfig[extract_task],
                template=LLM_TEMPLATE[config.ptemplate],
            )
        elif config.wtype == "hf":
            # Load model
            self.model, self.tokenizer = get_model(config=config)
            self.model.eval()

            self.infer_pipeline = HFWrapper(
                model=self.model,
                tokenizer=self.tokenizer,
                generation_config=GenerationConfig[extract_task],
                template=LLM_TEMPLATE[config.ptemplate],
            )

        elif config.wtype == "azuregpt":
            self.infer_pipeline = AzureGPTWrapper(
                engine=config.model_name,
                generation_config=GenerationConfig[extract_task],
            )
        elif config.wtype == "gemini":
            self.infer_pipeline = GeminiWrapper(
                model_name=config.model_name,
                generation_config=GenerationConfig[extract_task],
            )
        else:
            raise ValueError("Invalid wrapper type")

        self.config = config
        self.prompting_strategy = 0
        self.few_shot = False
        self.random_mtpc = False
        self.cot = False
        self.continue_infer_data = None
        # Metric pipeline configuration
        self.metric_pipeline = MetricPipeline()
        self.task_name, self.ds_name = None, None
        self.config.filepath = None

    def __call__(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        task = self.task.split("_")[0]

        if task == "question-answering":
            return self.__question_answering(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "summarization":
            return self.__summarization(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif "translation" in task:
            return self.__translation(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif "language-modelling" in task:
            return self.__language_modelling(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif "text-classification" in task:
            return self.__multiple_choice_text_classification(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "sentiment-analysis":
            return self.__multiple_choice_sentiment(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "toxicity-detection-ViCTSD":
            return self.__multiple_choice_toxicity(
                ds_wrapper, ds_loader, saving_fn, start_idx
            )
        elif task == "toxicity-detection-ViHSD":
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
        elif "reasoning" in task:
            return self.__reasoning(ds_wrapper, ds_loader, saving_fn, start_idx)
        else:
            raise NotImplementedError

    def __question_answering(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        if self.continue_infer_data is not None:
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
        idx = 0

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                [
                    {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
                    {
                        "role": "user",
                        "content": ds_wrapper.prompt["prompt"].format(
                            c,
                            q,
                        ),
                    },
                ]
                for c, q in zip(batch[ds_wrapper.context], batch[ds_wrapper.question])
            ]

            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
            predictions.extend(results)
            references.extend([x[0] for x in batch[ds_wrapper.answer]["text"]])
            generation_probs.extend([x.tolist() for x in logprobs])

            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {
                    "predictions": predictions,
                    "references": references,
                    "generation_probs": generation_probs,
                }
                saving_fn(generations)
                mean_result = self.metric_pipeline.run_mean(
                    generations, self.task_name, self.ds_name, self.config
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
        }
        mean_result = self.metric_pipeline.run_mean(
            generations, self.task_name, self.ds_name, self.config
        )
        std_result = self.metric_pipeline.run_std(
            generations, self.task_name, self.ds_name, self.config
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
        calibration_few_shot = []
        selected_sample = []
        if self.continue_infer_data is not None:
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
            calib_probs.extend(self.continue_infer_data["calibration_probs"])
        if self.few_shot:

            def preprocessing_a_record(rec):
                return [rec[ds_wrapper.question], rec[ds_wrapper.answer]]

            selected_sample_idx = list(
                random.sample(range(len(ds_wrapper.dataset_training)), 5)
            )
            selected_sample = [
                preprocessing_a_record(ds_wrapper.dataset_training[s])
                for s in selected_sample_idx
            ]

            original_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.prompt["prompt"],
                answer_format='{{ "answer": "{}", "confident_level": 1 }}',
            )
            calib_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.calibration_prompt["prompt"],
                answer_format="{}",
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
                for q in batch[ds_wrapper.question]
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
                for q in batch[ds_wrapper.question]
            ]

            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
            calibprob_batch, _ = self.infer_pipeline.compute_logprob_and_length(
                calib_prompts, batch[ds_wrapper.answer]
            )
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.answer]])
            generation_probs.extend([x.tolist() for x in logprobs])
            calib_probs.extend([x.tolist() for x in calibprob_batch])
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
                    generations, self.task_name, self.ds_name, self.config
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
            generations, self.task_name, self.ds_name, self.config
        )
        std_result = self.metric_pipeline.run_std(
            generations, self.task_name, self.ds_name, self.config
        )
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __summarization(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        original_documents = []
        predictions = []
        references = []
        generation_probs = []
        if self.continue_infer_data is not None:
            original_documents.extend(self.continue_infer_data["original_documents"])
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
        idx = 0

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                [
                    {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
                    {
                        "role": "user",
                        "content": ds_wrapper.prompt["prompt"].format(
                            document,
                        ),
                    },
                ]
                for document in batch[ds_wrapper.original_text]
            ]
            original_documents.extend([x for x in batch[ds_wrapper.original_text]])

            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.summarized_text]])
            generation_probs.extend([x.tolist() for x in logprobs])

            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {
                    "original_documents": original_documents,
                    "predictions": predictions,
                    "references": references,
                    "generation_probs": generation_probs,
                }
                saving_fn(generations)
                mean_result = self.metric_pipeline.run_mean(
                    generations, self.task_name, self.ds_name, self.config
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {
            "original_documents": original_documents,
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
        }
        mean_result = self.metric_pipeline.run_mean(
            generations, self.task_name, self.ds_name, self.config
        )
        std_result = self.metric_pipeline.run_std(
            generations, self.task_name, self.ds_name, self.config
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
        mapping = ["Tiêu cực", "Trung lập", "Tích cực"]
        if self.continue_infer_data is not None:
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
            option_probs.extend(self.continue_infer_data["option_probs"])
        if self.few_shot:

            def preprocessing_a_record(rec):
                return [rec[ds_wrapper.text], rec[ds_wrapper.label]]

            classes = unique(ds_wrapper.dataset_training[ds_wrapper.label])
            selected_sample = []
            for cl in classes:
                cl_samples = ds_wrapper.dataset_training.filter(
                    lambda r: r[ds_wrapper.label] == cl
                )
                selected_sample.append(
                    preprocessing_a_record(
                        cl_samples[random.randint(0, len(cl_samples))]
                    )
                )

            original_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.prompt["prompt"],
                answer_format='{{ "sentiment": {}, "confident_level": 1}}',
            )
            calib_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.calibration_prompt["prompt"],
                answer_format="{}",
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
                for c in batch[ds_wrapper.text]
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
                for c in batch[ds_wrapper.text]
            ]
            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
            num_choice = 3

            option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                calib_prompts * num_choice,
                [
                    mapping[choice] if self.prompting_strategy == 1 else str(choice)
                    for choice in range(num_choice)
                    for _ in range(len(prompts))
                ],
            )
            predictions.extend(results)
            references.extend([x.item() for x in batch[ds_wrapper.label]])
            generation_probs.extend([x.tolist() for x in logprobs])
            option_probs.extend(
                [
                    [
                        option_logprobs[i + opt * len(prompts)].tolist()
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
                    generations, self.task_name, self.ds_name, self.config
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
            generations, self.task_name, self.ds_name, self.config
        )
        std_result = self.metric_pipeline.run_std(
            generations, self.task_name, self.ds_name, self.config
        )
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __multiple_choice_text_classification(
        self, ds_wrapper, ds_loader, saving_fn, start_idx=0
    ):
        sub_task = self.task.split("-")[-1]
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
        if sub_task == "vsmec":
            num_choice = 7

        elif sub_task == "atis":
            num_choice = 17
        else:
            raise ValueError("Invalid sub task")

        if self.few_shot:

            def preprocessing_a_record(rec):
                return [rec[ds_wrapper.text], rec[ds_wrapper.label]]

            def format_original_fewshot0(rec):
                return f"""Khách: "{rec[ds_wrapper.text]}"\nBot:[/INST] {{ "{"emotion" if sub_task == "vsmec" else "tag"}": {rec[ds_wrapper.label]}, "confident_level": 1}} </s><s>[INST]\n"""

            def format_original_fewshot1(rec):
                return f"""Khách: "{rec[ds_wrapper.text]}"\nBot: {{ "{"emotion" if sub_task == "vsmec" else "tag"}": {rec[ds_wrapper.label]}, "confident_level": 1}}\n"""

            def format_calib_fewshot(rec):
                return f"""Khách: "{rec[ds_wrapper.text]}"\nBot:[/INST] {rec[ds_wrapper.label]} </s><s>[INST]\n"""

            classes = (
                unique(ds_wrapper.dataset_training[ds_wrapper.label])
                if sub_task == "vsmec"
                else unique(column(ds_wrapper.dataset_training[ds_wrapper.label], 0))
            )

            selected_sample = []
            for cl in classes:
                cl_samples = ds_wrapper.dataset_training.filter(
                    lambda r: (
                        r[ds_wrapper.label] == cl
                        if sub_task == "vsmec"
                        else r[ds_wrapper.label][0] == cl
                    )
                )
                selected_sample.append(
                    cl_samples[random.randint(0, len(cl_samples) - 1)]
                )

            if sub_task == "atis":
                for x in range(len(selected_sample)):
                    selected_sample[x][ds_wrapper.label] = selected_sample[x][
                        ds_wrapper.label
                    ][0]
            selected_sample = [preprocessing_a_record(x) for x in selected_sample]
            field_name = "emotion" if sub_task == "vsmec" else "tag"
            original_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.prompt["prompt"],
                answer_format='{{  "' + field_name + '": {}, "confident_level": 1}}',
            )
            calib_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.calibration_prompt["prompt"],
                answer_format="{}",
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
                for c in batch[ds_wrapper.text]
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
                for c in batch[ds_wrapper.text]
            ]

            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)

            option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                calib_prompts * num_choice,
                [
                    str(choice)
                    for choice in range(num_choice)
                    for _ in range(len(prompts))
                ],
            )
            predictions.extend(results)
            references.extend(
                [
                    eval(x) if type(x) is str else x.item()
                    for x in batch[ds_wrapper.label]
                ]
            )
            generation_probs.extend([x.tolist() for x in logprobs])
            option_probs.extend(
                [
                    [
                        option_logprobs[i + opt * len(prompts)].tolist()
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
                    generations, self.task_name, self.ds_name, self.config
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
            generations, self.task_name, self.ds_name, self.config
        )
        std_result = self.metric_pipeline.run_std(
            generations, self.task_name, self.ds_name, self.config
        )
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __multiple_choice_toxicity(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        sub_task = self.task.split("-")[2]
        predictions = []
        references = []
        generation_probs = []
        option_probs = []
        idx = 0
        original_few_shot = []
        calib_few_shot = []
        selected_sample = []
        if self.continue_infer_data is not None:
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
            option_probs.extend(self.continue_infer_data["option_probs"])
        if self.few_shot:

            def preprocessing_a_record(rec):
                return [rec[ds_wrapper.text], rec[ds_wrapper.label]]

            classes = unique(ds_wrapper.dataset_training[ds_wrapper.label])
            selected_sample = []
            for cl in classes:
                cl_samples = ds_wrapper.dataset_training.filter(
                    lambda r: r[ds_wrapper.label] == cl
                )
                selected_sample.append(
                    preprocessing_a_record(
                        cl_samples[random.randint(0, len(cl_samples))]
                    )
                )

            original_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.prompt["prompt"],
                answer_format='{{ "toxicity_level": {}, "confident_level": 1}}',
            )
            calib_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.calibration_prompt["prompt"],
                answer_format="{}",
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
                for c in batch[ds_wrapper.text]
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
                for c in batch[ds_wrapper.text]
            ]
            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
            num_choice = 2 if sub_task == "ViCTSD" else 3

            option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                calib_prompts * num_choice,
                [
                    str(choice)
                    for choice in range(num_choice)
                    for _ in range(len(prompts))
                ],
            )
            predictions.extend(results)
            references.extend([x.item() for x in batch[ds_wrapper.label]])
            generation_probs.extend([x.tolist() for x in logprobs])
            option_probs.extend(
                [
                    [
                        option_logprobs[i + opt * len(prompts)].tolist()
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
                    generations, self.task_name, self.ds_name, self.config
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
            generations, self.task_name, self.ds_name, self.config
        )
        std_result = self.metric_pipeline.run_std(
            generations, self.task_name, self.ds_name, self.config
        )
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __multiple_choice(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        def format_list_ans(ans_list):
            return "\n".join(
                list(
                    map(
                        lambda ans: f"{chr(ans[0]+65)}: ''' {ans[1]} '''",
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
        alphabet2idx = {chr(i + 65): i for i in range(26)}
        if self.continue_infer_data is not None:
            predictions.extend(self.continue_infer_data["predictions"])
            references.extend(self.continue_infer_data["references"])
            generation_probs.extend(self.continue_infer_data["generation_probs"])
            option_probs.extend(self.continue_infer_data["option_probs"])
            option_order_all.extend(self.continue_infer_data["option_orders"])

        if self.few_shot:

            def preprocessing_a_record(rec):
                return [
                    rec[ds_wrapper.context],
                    rec[ds_wrapper.question],
                    format_list_ans(rec[ds_wrapper.options]),
                    rec[ds_wrapper.answer],
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
                answer_format='{{ "choice": "{}", "confident_level": 1 }}',
            )
            calib_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.calibration_prompt["prompt"],
                answer_format="{}",
            )

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = []
            calib_prompts = []
            remap_order_batch = []
            for o_idx, cq in enumerate(
                zip(batch[ds_wrapper.context], batch[ds_wrapper.question])
            ):
                c = cq[0]
                q = cq[1]
                opts = column(batch[ds_wrapper.options], o_idx)
                order_shuffle = list(range(len(opts)))
                if self.random_mtpc:
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
                calib_prompts * 4,
                [chr(choice + 65) for choice in range(4) for _ in range(len(prompts))],
            )
            opt_calib_out = [
                [option_logprobs[i + opt * len(prompts)].tolist() for opt in range(4)]
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
                    chr(remap.index(alphabet2idx[x]) + 65)
                    for x, remap in zip(batch[ds_wrapper.answer], remap_order_batch)
                ]
            )

            generation_probs.extend([x.tolist() for x in logprobs])
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
                    generations, self.task_name, self.ds_name, self.config
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
            generations, self.task_name, self.ds_name, self.config
        )
        std_result = self.metric_pipeline.run_std(
            generations, self.task_name, self.ds_name, self.config
        )
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __language_modelling(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
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
                return [rec[ds_wrapper.source], rec[ds_wrapper.target]]

            selected_sample = [
                preprocessing_a_record(s) for s in ds_wrapper.dataset_training
            ]
            original_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.prompt["prompt"],
                answer_format="{}",
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
                for c in batch[ds_wrapper.source]
            ]

            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.target]])
            generation_probs.extend([x.tolist() for x in logprobs])

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
                    generations, self.task_name, self.ds_name, self.config
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "fewshot": selected_sample,
        }
        mean_result = self.metric_pipeline.run_mean(
            generations, self.task_name, self.ds_name, self.config
        )
        std_result = self.metric_pipeline.run_std(
            generations, self.task_name, self.ds_name, self.config
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
                    rec[ds_wrapper.passage],
                    rec[ds_wrapper.query],
                    rec[ds_wrapper.answer],
                ]

            random_sample = list(random.sample(list(ds_wrapper.dataset_training), 1))[0]
            # random_batch_passages = random_sample[ds_wrapper.passage]
            # if sub_task == "mmarco":
            #     ref_passage_id = random_sample[ds_wrapper.answer][0]
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
            #     ref_passage_id = random_sample[ds_wrapper.answer][0]
            #     ref_passage_idx = random_batch_passages["id"].index(
            #         ref_passage_id)
            #     rnd_passage_id = random_sample[ds_wrapper.answer][-1]
            #     rnd_passage_idx = batch_passages["id"].index(rnd_passage_id)

            first_sample = {
                "passages": random_sample["positive"],
                "query": random_sample[ds_wrapper.query],
                "references": "Yes",
            }
            second_sample = {
                "passages": random_sample["negative"],
                "query": random_sample[ds_wrapper.query],
                "references": "No",
            }

            selected_sample = [
                preprocessing_a_record(s) for s in [first_sample, second_sample]
            ]
            original_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.prompt["prompt"],
                answer_format='{{ "answer": "{}" }}',
            )
            calib_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.calibration_prompt["prompt"],
                answer_format="{}",
            )

        BATCH_PASSAGE_SIZE = 10
        # Create few-shot strings
        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue
            for query_with_a_batch_passages in range(len(batch[ds_wrapper.id])):
                query_id = batch[ds_wrapper.id][query_with_a_batch_passages]
                query = batch[ds_wrapper.query][query_with_a_batch_passages]
                try:
                    ref_passage_id = batch[ds_wrapper.answer][0].tolist()[
                        query_with_a_batch_passages
                    ]
                except:
                    if len(list(batch[ds_wrapper.answer])) < 1:
                        continue
                    ref_passage_id = list(batch[ds_wrapper.answer][0])[
                        query_with_a_batch_passages
                    ]
                batch_passages = batch[ds_wrapper.passage]

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
                                "generation_probs": y.tolist(),
                                "calib_probs": [
                                    option_logprobs[q + opt * len(prompts)].tolist()
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
                    generations, self.task_name, self.ds_name, self.config
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {"fewshot": selected_sample, "predictions": predictions}
        mean_result = self.metric_pipeline.run_mean(
            generations, self.task_name, self.ds_name, self.config
        )
        std_result = self.metric_pipeline.run_std(
            generations, self.task_name, self.ds_name, self.config
        )
        final_result = {"mean": mean_result, "std": std_result}
        saving_fn(generations, final_result)

    def __reasoning(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        calib_probs = []
        math_problem_type = []
        sub_task = self.task.split("-")[-1]
        idx = 0
        original_few_shot = []
        calib_few_shot = []
        selected_sample = []
        if not self.cot and sub_task == "math":
            target = ds_wrapper.short_target
        else:
            target = ds_wrapper.target
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
                return [rec[ds_wrapper.source], rec[target]]

            selected_sample = [
                preprocessing_a_record(s)
                for s in list(random.sample(list(ds_wrapper.dataset_training), 5))
            ]
            original_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.prompt["prompt"],
                answer_format='{{ "answer": "{}", "confident_level": 1 }}',
            )
            calib_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.calibration_prompt["prompt"],
                answer_format="{}",
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
                for rule in batch[ds_wrapper.source]
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
                for rule in batch[ds_wrapper.source]
            ]

            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
            calibprob_batch, _ = self.infer_pipeline.compute_logprob_and_length(
                calib_prompts, batch[target]
            )
            predictions.extend(results)
            references.extend([x for x in batch[target]])
            generation_probs.extend([x.tolist() for x in logprobs])
            calib_probs.extend([x.tolist() for x in calibprob_batch])
            if sub_task == "math":
                math_problem_type.extend([x for x in batch[ds_wrapper.type]])
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
                if sub_task == "math":
                    generations["math_problem_type"] = math_problem_type
                saving_fn(generations)
                mean_result = self.metric_pipeline.run_mean(
                    generations, self.task_name, self.ds_name, self.config
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "calibration_probs": calib_probs,
            "fewshot": selected_sample,
        }
        if sub_task == "math":
            generations["math_problem_type"] = math_problem_type
        mean_result = self.metric_pipeline.run_mean(
            generations, self.task_name, self.ds_name, self.config
        )
        mean_result = self.metric_pipeline.run_mean(
            generations, self.task_name, self.ds_name, self.config
        )
        std_result = self.metric_pipeline.run_std(
            generations, self.task_name, self.ds_name, self.config
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
                    rec[ds_wrapper.source_language],
                    rec[ds_wrapper.target_language],
                ]

            selected_sample = [
                preprocessing_a_record(s)
                for s in list(random.sample(list(ds_wrapper.dataset_training), 5))
            ]
            original_few_shot = format_fewshot(
                selected_sample,
                query_format=ds_wrapper.prompt["prompt"],
                answer_format='{{ "translation": "{}" }}',
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
                for document in batch[ds_wrapper.source_language]
            ]

            results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.target_language]])
            generation_probs.extend([x.tolist() for x in logprobs])

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
                    generations, self.task_name, self.ds_name, self.config
                )
                print(f"Results of {idx} batches: ", mean_result)

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "fewshot": selected_sample,
        }
        mean_result = self.metric_pipeline.run_mean(
            generations, self.task_name, self.ds_name, self.config
        )
        std_result = self.metric_pipeline.run_std(
            generations, self.task_name, self.ds_name, self.config
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
        random_mtpc=False,
        cot=False,
        prompting_strategy=0,
        continue_infer=None,
    ):
        self.generation_results_file = generation_results_file
        filename = os.path.basename(self.generation_results_file)
        self.task_name, self.ds_name, _, _, _ = info_from_filename(filename)
        self.config.filepath = generation_results_file
        self.continue_infer_data = continue_infer
        self.prompting_strategy = prompting_strategy
        self.few_shot = few_shot
        self.random_mtpc = random_mtpc
        self.cot = cot
        with torch.no_grad():
            results = self(ds_wrapper, ds_loader, saving_fn, start_idx)
        return results
