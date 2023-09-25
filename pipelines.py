from utils import *
import ast

import torch
from generation_config import GenerationConfig

from tqdm import tqdm


class InferPipeline:
    def __init__(self, model, tokenizer, generation_config):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.few_shot_flag = False
        self.random_mtpc = False

    def __call__(self, prompts, return_probs=False):
        generations = []
        generations_probs = []
        num_generated_tokens = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt, return_tensors="pt").to(self.model.device)
            generate_dict = self.model.generate(
                inputs.input_ids,
                output_scores=True,
                return_dict_in_generate=True,
                **self.generation_config,
            )

            num_generated_token = len(generate_dict.scores)
            num_generated_tokens.append(num_generated_token)
            generated_tokens = generate_dict.sequences[:, -
                                                       num_generated_token:]

            generation = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            generations.extend(generation)

            if return_probs:
                generation_probs = self.model.compute_transition_scores(
                    sequences=generated_tokens,
                    scores=generate_dict.scores,
                    normalize_logits=True,
                )
                generations_probs.extend(generation_probs.cpu().numpy())

        return generations, generations_probs, num_generated_tokens

    def compute_logprob_and_length(self, prompts, completions):
        completions_num_tokens = []
        completions_logprobs = []

        for prompt, completion in zip(prompts, completions):
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt").to(
                self.model.device
            )  # <s> [tokens]
            # Actual number of tokens in completion (without `<s>`)
            prompt_num_tokens = prompt_tokens.input_ids.shape[1] - 1

            completion_tokens = self.tokenizer(completion, return_tensors="pt").to(
                self.model.device
            )  # <s> [tokens]
            # Actual number of tokens in completion (without `<s>`)
            completion_num_tokens = completion_tokens.input_ids.shape[1] - 1
            completions_num_tokens.append(completion_num_tokens)

            inputs = torch.concatenate(
                (prompt_tokens.input_ids,
                 completion_tokens.input_ids[:, 1:]), dim=-1
            )
            outputs = self.model(inputs)  # [input_tokens] [next_token]

            logits = outputs.logits[
                :, prompt_num_tokens: prompt_num_tokens + completion_num_tokens
            ]
            logprobs = logits.log_softmax(dim=-1)
            # >>> batch_size, sequence_length, vocab_size

            logprobs = logprobs.gather(
                dim=-1, index=completion_tokens.input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            # >>> batch_size, sequence_length
            completions_logprobs.append(logprobs.cpu().numpy())

        return completions_logprobs, completions_num_tokens


class EvalPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        extract_task = self.task.split("_")[0]
        self.model = model
        self.tokenizer = tokenizer
        self.infer_pipeline = InferPipeline(
            model=model,
            tokenizer=tokenizer,
            generation_config=GenerationConfig[extract_task],
        )

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
        idx = 0

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(
                    context=c,
                    question=q,
                )
                for c, q in zip(batch[ds_wrapper.context], batch[ds_wrapper.question])
            ]

            results, logprobs, _ = self.infer_pipeline(
                prompts, return_probs=True)
            predictions.extend(results)
            references.extend([x[0] for x in batch[ds_wrapper.answer]["text"]])
            generation_probs.extend([x for x in logprobs])

            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {
                    "predictions": predictions,
                    "references": references,
                    "generation_probs": generation_probs,
                }
                saving_fn(generations)

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
        }
        saving_fn(generations)

    def __question_answering_without_context(
        self, ds_wrapper, ds_loader, saving_fn, start_idx=0
    ):
        predictions = []
        references = []
        generation_probs = []
        calib_probs = []
        idx = 0
        original_few_shot = ""
        selected_sample = []
        if self.few_shot_flag:

            def format_original_fewshot0(rec):
                return f"""Câu hỏi: {rec[ds_wrapper.question]}\nCâu trả lời:[/INST] {rec[ds_wrapper.answer]} </s><s>[INST]\n"""

            def format_original_fewshot1(rec):
                return f"""Câu hỏi: {rec[ds_wrapper.question]}\nTrả lời: {rec[ds_wrapper.answer]}\n\n"""

            selected_sample_idx = list(
                random.sample(range(len(ds_wrapper.dataset_training)), 5)
            )
            selected_sample = [
                ds_wrapper.dataset_training[s] for s in selected_sample_idx
            ]

            original_few_shot = "".join(
                list(
                    map(
                        format_original_fewshot1
                        if self.prompting_strategy == 1
                        else format_original_fewshot1,
                        selected_sample,
                    )
                )
            )

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(
                    few_shot=original_few_shot,
                    question=q,
                )
                for q in batch[ds_wrapper.question]
            ]
            print(prompts[0])
            results, logprobs, _ = self.infer_pipeline(
                prompts, return_probs=True)
            calibprob_batch, _ = self.infer_pipeline.compute_logprob_and_length(
                prompts, batch[ds_wrapper.answer]
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

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "calibration_probs": calib_probs,
            "fewshot": selected_sample,
        }
        saving_fn(generations)

    def __summarization(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        original_documents = []
        predictions = []
        references = []
        generation_probs = []
        idx = 0

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(document=document)
                for document in batch[ds_wrapper.original_text]
            ]
            original_documents.extend(
                [x for x in batch[ds_wrapper.original_text]])

            results, logprobs, _ = self.infer_pipeline(
                prompts, return_probs=True)
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.summarized_text]])
            generation_probs.extend([x for x in logprobs])

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

        generations = {
            "original_documents": original_documents,
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
        }
        saving_fn(generations)

    def __multiple_choice_sentiment(
        self, ds_wrapper, ds_loader, saving_fn, start_idx=0
    ):
        predictions = []
        references = []
        generation_probs = []
        option_probs = []
        idx = 0
        original_few_shot = ""
        calib_few_shot = ""
        selected_sample = []
        mapping = ["Tiêu cực", "Trung lập", "Tích cực"]
        if self.few_shot_flag:

            def format_original_fewshot0(rec):
                return f"""Khách: "{rec[ds_wrapper.text]}"\nBot:[/INST] {{ "sentiment": {rec[ds_wrapper.label]}, "confident_level": 1}} </s><s>[INST]\n"""

            def format_original_fewshot1(rec):
                return f"""Đoạn văn: {rec[ds_wrapper.text]}\nQuan điểm: {mapping[rec[ds_wrapper.label]]}\n\n"""

            def format_calib_fewshot(rec):
                return f"""Khách: "{rec[ds_wrapper.text]}"\nBot:[/INST] {rec[ds_wrapper.label]} </s><s>[INST]\n"""

            classes = unique(ds_wrapper.dataset_training[ds_wrapper.label])
            selected_sample = []
            for cl in classes:
                cl_samples = ds_wrapper.dataset_training.filter(
                    lambda r: r[ds_wrapper.label] == cl
                )
                selected_sample.append(
                    cl_samples[random.randint(0, len(cl_samples))])

            original_few_shot = "".join(
                list(
                    map(
                        format_original_fewshot1
                        if self.prompting_strategy == 1
                        else format_original_fewshot0,
                        selected_sample,
                    )
                )
            )
            calib_few_shot = "".join(
                list(
                    map(
                        format_original_fewshot1
                        if self.prompting_strategy == 1
                        else format_calib_fewshot,
                        selected_sample,
                    )
                )
            )

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(context=c, few_shot=original_few_shot)
                for c in batch[ds_wrapper.text]
            ]
            calib_prompts = [
                ds_wrapper.calibration_prompt.format(
                    context=c, few_shot=calib_few_shot)
                for c in batch[ds_wrapper.text]
            ]
            results, logprobs, _ = self.infer_pipeline(
                prompts, return_probs=True)
            num_choice = 3

            option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                calib_prompts * num_choice,
                [
                    mapping[choice] if self.prompting_strategy == 1 else str(
                        choice)
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

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "option_probs": option_probs,
            "fewshot": selected_sample,
        }

        saving_fn(generations)

    def __multiple_choice_text_classification(
        self, ds_wrapper, ds_loader, saving_fn, start_idx=0
    ):
        sub_task = self.task.split("-")[2]
        predictions = []
        references = []
        generation_probs = []
        option_probs = []
        idx = 0
        original_few_shot = ""
        calib_few_shot = ""
        selected_sample = []
        if sub_task == "vsmec":
            mapping = [
                "Sadness",
                "Surprise",
                "Disgust",
                "Fear",
                "Anger",
                "Other",
                "Enjoyment",
            ]
        else:
            mapping = [
                "flight",
                "airfare",
                "ground_service",
                "day_name",
                "meal",
                "airport",
                "airline",
                "flight_time",
                "city",
                "ground_fare",
                "quantity",
                "abbreviation",
                "distance",
                "aircraft",
                "capacity",
                "flight_no",
                "restriction",
            ]
        if self.few_shot_flag:

            def format_original_fewshot0(rec):
                return f"""Khách: "{rec[ds_wrapper.text]}"\nBot:[/INST] {{ {"emotion" if sub_task == "vsmec" else "tag"}: {rec[ds_wrapper.label]}, "confident_level": 1}} </s><s>[INST]\n"""

            def format_original_fewshot1(rec):
                return f"""Đoạn văn: {rec[ds_wrapper.text]}\nNhãn: {mapping[rec[ds_wrapper.label]]}\n\n"""

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
                    lambda r: r[ds_wrapper.label] == cl
                    if sub_task == "vsmec"
                    else r[ds_wrapper.label][0] == cl
                )
                selected_sample.append(
                    cl_samples[random.randint(0, len(cl_samples))])

            if sub_task == "atis":
                for x in range(len(selected_sample)):
                    selected_sample[x][ds_wrapper.label] = selected_sample[x][
                        ds_wrapper.label
                    ][0]

            original_few_shot = "".join(
                list(
                    map(
                        format_original_fewshot1
                        if self.prompting_strategy == 1
                        else format_original_fewshot0,
                        selected_sample,
                    )
                )
            )
            calib_few_shot = "".join(
                list(
                    map(
                        format_original_fewshot1
                        if self.prompting_strategy == 1
                        else format_calib_fewshot,
                        selected_sample,
                    )
                )
            )

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(
                    few_shot=original_few_shot,
                    context=c,
                )
                for c in batch[ds_wrapper.text]
            ]

            calib_prompts = [
                ds_wrapper.calibration_prompt.format(
                    few_shot=calib_few_shot,
                    context=c,
                )
                for c in batch[ds_wrapper.text]
            ]

            results, logprobs, _ = self.infer_pipeline(
                prompts, return_probs=True)

            num_choice = 7 if sub_task == "vsmec" else 17
            option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                calib_prompts * num_choice,
                [
                    mapping[choice] if self.prompting_strategy == 1 else str(
                        choice)
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

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "option_probs": option_probs,
            "fewshot": selected_sample,
        }
        saving_fn(generations)

    def __multiple_choice_toxicity(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        sub_task = self.task.split("-")[2]
        predictions = []
        references = []
        generation_probs = []
        option_probs = []
        idx = 0
        original_few_shot = ""
        calib_few_shot = ""
        selected_sample = []
        if self.few_shot_flag:

            def format_original_fewshot(rec):
                return f"""Khách: "{rec[ds_wrapper.text]}"\nBot:[/INST] {{ "toxic_level": {rec[ds_wrapper.label]}, "confident_level": 1}} </s><s>[INST]\n"""

            def format_calib_fewshot(rec):
                return f"""Khách: "{rec[ds_wrapper.text]}"\nBot:[/INST] {rec[ds_wrapper.label]} </s><s>[INST]\n"""

            classes = unique(ds_wrapper.dataset_training[ds_wrapper.label])
            selected_sample = []
            for cl in classes:
                cl_samples = ds_wrapper.dataset_training.filter(
                    lambda r: r[ds_wrapper.label] == cl
                )
                selected_sample.append(
                    cl_samples[random.randint(0, len(cl_samples))])

            original_few_shot = "".join(
                list(map(format_original_fewshot, selected_sample))
            )
            calib_few_shot = "".join(
                list(map(format_calib_fewshot, selected_sample)))

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(
                    few_shot=original_few_shot,
                    context=c,
                )
                for c in batch[ds_wrapper.text]
            ]

            calib_prompts = [
                ds_wrapper.calibration_prompt.format(
                    few_shot=calib_few_shot,
                    context=c,
                )
                for c in batch[ds_wrapper.text]
            ]
            results, logprobs, _ = self.infer_pipeline(
                prompts, return_probs=True)
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

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "option_probs": option_probs,
            "fewshot": selected_sample,
        }
        saving_fn(generations)

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
        original_few_shot = ""
        calib_few_shot = ""
        option_order_all = []
        selected_sample = []
        alphabet2idx = {chr(i + 65): i for i in range(26)}
        if self.few_shot_flag:

            def format_original_fewshot(rec):
                return f"""Ngữ cảnh: ''' {rec[ds_wrapper.context]} '''\nCâu hỏi: Hãy lựa chọn đáp án đúng. {rec[ds_wrapper.question]}\n{format_list_ans(rec[ds_wrapper.options])}\n\nCâu trả lời:[/INST] {{ "choice": "{rec[ds_wrapper.answer]}", "confident_level": 1 }} </s><s>[INST]\n"""

            def format_calib_fewshot(rec):
                return f"""Ngữ cảnh: ''' {rec[ds_wrapper.context]} \nCâu hỏi: Hãy lựa chọn đáp án đúng. {rec[ds_wrapper.question]}\n{format_list_ans(rec[ds_wrapper.options])}\n\nCâu trả lời:[/INST] {rec[ds_wrapper.answer]} </s><s>[INST]\n"""

            selected_sample_idx = list(
                random.sample(range(len(ds_wrapper.dataset_training)), 2)
            )
            selected_sample = [
                ds_wrapper.dataset_training[s] for s in selected_sample_idx
            ]

            original_few_shot = "".join(
                list(map(format_original_fewshot, selected_sample))
            )
            calib_few_shot = "".join(
                list(map(format_calib_fewshot, selected_sample)))

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = []
            calib_prompts = []
            remap_order_batch = []
            for o_idx, cq in enumerate(zip(
                batch[ds_wrapper.context],
                batch[ds_wrapper.question]
            )):
                c = cq[0]
                q = cq[1]
                opts = column(batch[ds_wrapper.options], o_idx)
                order_shuffle = (
                    random.shuffle(list(range(len(opts))))
                    if self.random_mtpc
                    else list(range(len(opts)))
                )
                remap_order_batch.append(order_shuffle)
                new_opts = [opts[i] for i in order_shuffle]
                prompts.append(
                    ds_wrapper.prompt.format(
                        few_shot=original_few_shot,
                        context=c,
                        question=q,
                        list_ans=format_list_ans(new_opts),
                    )
                )
                calib_prompts.append(
                    ds_wrapper.calibration_prompt.format(
                        few_shot=calib_few_shot,
                        context=c,
                        question=q,
                        list_ans=format_list_ans(new_opts),
                    )
                )
            print(prompts[0])
            results, logprobs, _ = self.infer_pipeline(
                prompts, return_probs=True)
            option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                calib_prompts * 4,
                [chr(choice + 65) for choice in range(4)
                 for _ in range(len(prompts))],
            )
            opt_calib_out = [
                [option_logprobs[i + opt * len(prompts)].tolist()
                 for opt in range(4)]
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
                    "references": references,
                    "generation_probs": generation_probs,
                    "option_probs": option_probs,
                    "option_orders": option_order_all,
                    "fewshot": selected_sample,
                }
                saving_fn(generations)

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "option_probs": option_probs,
            "option_orders": option_order_all,
            "fewshot": selected_sample,
        }

        saving_fn(generations)

    def __language_modelling(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        idx = 0
        original_few_shot = ""
        selected_sample = []
        if self.few_shot_flag:

            def format_original_fewshot(rec):
                return f"""Khách: "{rec[ds_wrapper.source]}"\nBot:[/INST] {rec[ds_wrapper.target]} </s><s>[INST]\n"""

            selected_sample = ds_wrapper.dataset_training
            original_few_shot = "".join(
                list(map(format_original_fewshot, selected_sample))
            )

        # Create few-shot strings
        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(few_shot=original_few_shot, context=c)
                for c in batch[ds_wrapper.source]
            ]

            results, logprobs, _ = self.infer_pipeline(
                prompts, return_probs=True)
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

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "fewshot": selected_sample,
        }
        saving_fn(generations)

    def __information_retrieval(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        sub_task = self.task.split("_")[1]
        idx = 0
        original_few_shot = ""
        selected_sample = []
        if self.few_shot_flag:

            def format_original_fewshot(rec):
                return f"""Văn bản: ''' {rec["passage"]} '''\nCâu hỏi: ''' {rec["query"]} '''\n"Văn bản trên có thể hỗ trợ trả lời câu hỏi không?. Đưa ra câu trả lời của bạn dưới dạng JSON với định dạng là ```json {{ \"answer\": ` \"Yes\" or \"No\" `}} ```\nBot:[/INST] {{ "answer": "{rec["answer"]}" }} </s><s>[INST]\n"""

            def format_calib_fewshot(rec):
                return f"""Văn bản: ''' {rec["passage"]} '''\nCâu hỏi: ''' {rec["query"]} '''\n"Văn bản trên có thể hỗ trợ trả lời câu hỏi không?\nBot:[/INST] {rec["answer"]} </s><s>[INST]\n"""

            random_sample = list(random.sample(
                list(ds_wrapper.dataset_training), 1))[0]
            random_batch_passages = random_sample[ds_wrapper.passage]
            if sub_task == "mmarco":
                ref_passage_id = random_sample[ds_wrapper.answer][0]
                ref_passage_idx = random_batch_passages["id"].index(
                    ref_passage_id)
                rnd_passage_idx = random.choice(
                    [
                        i
                        for i in range(len(random_batch_passages["id"]))
                        if i != ref_passage_idx
                    ]
                )

            else:
                ref_passage_id = random_sample[ds_wrapper.answer][0]
                ref_passage_idx = random_batch_passages["id"].index(
                    ref_passage_id)
                rnd_passage_id = random_sample[ds_wrapper.answer][-1]
                rnd_passage_idx = batch_passages["id"].index(rnd_passage_id)

            first_sample = {
                "query": random_sample[ds_wrapper.query],
                "passage": random_batch_passages["passage"][ref_passage_idx],
                "answer": "Yes",
            }
            second_sample = {
                "query": random_sample[ds_wrapper.query],
                "passage": random_batch_passages["passage"][rnd_passage_idx],
                "answer": "No",
            }

            selected_sample = [first_sample, second_sample]

            original_few_shot = "".join(
                list(map(format_original_fewshot, selected_sample))
            )
            calib_few_shot = "".join(
                list(map(format_calib_fewshot, selected_sample)))
        BATCH_PASSAGE_SIZE = 5
        # Create few-shot strings
        # for batch in tqdm(ds_loader):
        #     if idx < start_idx:
        #         idx += 1
        #         continue
        #             prompts = []
        #             log_data = []

        #             for query_with_a_batch_passages in range(len(batch[ds_wrapper.id])):
        #                 query_id = batch[ds_wrapper.id][query_with_a_batch_passages]
        #                 query = batch[ds_wrapper.query][query_with_a_batch_passages]
        #                 batch_passages = batch[ds_wrapper.passage]
        #                 try:
        #                     if sub_task == "mmarco":
        #                         ref_passage_id = batch[ds_wrapper.answer][0].tolist()[query_with_a_batch_passages]
        #                         ref_passage_idx = column(batch_passages['id'], query_with_a_batch_passages).index(ref_passage_id)
        #                         rnd_passage_idx = random.choice([i for i in range(len(column(batch_passages['id'], query_with_a_batch_passages))) if i != ref_passage_idx])

        #                     else:
        #                         ref_passage_id = batch[ds_wrapper.answer][query_with_a_batch_passages].tolist()[0]
        #                         ref_passage_idx = batch_passages['id'][query_with_a_batch_passages].tolist().index(ref_passage_id)
        #                         rnd_passage_id = batch[ds_wrapper.answer][query_with_a_batch_passages].tolist()[-1].item()
        #                         rnd_passage_idx = batch_passages['id'][query_with_a_batch_passages].tolist().index(rnd_passage_id)
        #                 except:
        #                     continue
        #                 list_passage_idx = [ref_passage_idx, rnd_passage_idx]
        #                 label_passage = [1, 0]
        #                 psgl = column(batch_passages['passage'], query_with_a_batch_passages)
        #                 prompts.extend([
        #                         ds_wrapper.prompt.format(
        #                             few_shot=original_few_shot, passage=psgl[p], question=query
        #                         )
        #                         for p in list_passage_idx
        #                     ])
        #                 log_data.extend([
        #                     {
        #                         "query_id": query_id.item(),
        #                         "query": query,
        #                         "passage_id": column(batch_passages['id'], query_with_a_batch_passages)[p].item(),
        #                         "passage": column(batch_passages['passage'], query_with_a_batch_passages)[p],
        #                         "label": label_passage[i]
        #                     }
        #                     for i, p in enumerate(list_passage_idx)
        #                 ])
        #             for psg in range(0, len(prompts), BATCH_PASSAGE_SIZE):
        #                 results, logprobs, _ = self.infer_pipeline(
        #                         prompts[psg : psg + BATCH_PASSAGE_SIZE], return_probs=True
        #                 )
        #                 for l in range(psg, psg+BATCH_PASSAGE_SIZE if psg+BATCH_PASSAGE_SIZE < len(prompts) else len(prompts)):
        #                    log_data[l]["prediction"] = results[l-psg]
        #                    log_data[l]["generation_probs"] = logprobs[l-psg].tolist()

        #             predictions.extend(log_data)
        #             idx += 1
        #             if idx % 100 == 0:
        #                 print(f"Saving results of {idx} batches")

        #                 generations = {"fewshot": selected_sample,
        #                                "prediction": predictions}
        #                 saving_fn(generations)

        #         generations = {"fewshot": selected_sample, "prediction": predictions}
        #         saving_fn(generations)

        BATCH_PASSAGE_SIZE = 5
        # Create few-shot strings
        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue
            for query_with_a_batch_passages in range(len(batch[ds_wrapper.id])):
                query_id = batch[ds_wrapper.id][query_with_a_batch_passages]
                query = batch[ds_wrapper.query][query_with_a_batch_passages]
                ref_passage_id = batch[ds_wrapper.answer][0].tolist()[
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
                        ds_wrapper.prompt.format(
                            few_shot=original_few_shot, passage=p, question=query
                        )
                        for p in top30_passages[psg: psg + BATCH_PASSAGE_SIZE]
                    ]
                    calib_prompts = [
                        ds_wrapper.calibration_prompt.format(
                            few_shot=calib_few_shot, passage=p, question=query
                        )
                        for p in top30_passages[psg: psg + BATCH_PASSAGE_SIZE]
                    ]
                    results, logprobs, _ = self.infer_pipeline(
                        prompts, return_probs=True
                    )

                    option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                        calib_prompts * len(prompts),
                        [
                            choice
                            for choice in ["Yes", "No"]
                            for _ in range(len(prompts))
                        ],
                    )
                    save_each_prompt = list(
                        map(
                            lambda x, y, z, t, q: {
                                "query_id": query_id.item(),
                                "query": query,
                                "passage_id": z.item(),
                                "passage": t,
                                "label": int(z.item() == ref_passage_id),
                                "prediction": x,
                                "generation_probs": y.tolist(),
                                "calib_probs": [
                                    option_logprobs[q + opt *
                                                    len(prompts)].tolist()
                                    for opt in range(2)
                                ],
                            },
                            results,
                            logprobs,
                            top30_passage_ids,
                            top30_passages,
                            range(len(prompts)),
                        )
                    )
                    predictions.extend(save_each_prompt)

            idx += 1

            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")

                generations = {"fewshot": selected_sample,
                               "prediction": predictions}
                saving_fn(generations)
        generations = {"fewshot": selected_sample, "prediction": predictions}
        saving_fn(generations)

    def __reasoning(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        calib_probs = []
        math_problem_type = []
        sub_task = self.task.split("-")[-1]
        idx = 0
        original_few_shot = ""
        calib_few_shot = ""
        selected_sample = []
        if self.few_shot_flag:

            def format_original_fewshot0(rec):
                return f"""{"Quy luật" if sub_task != "math" else "Bài toán"}: ```\n{rec[ds_wrapper.source]}\n```\n{"Kết quả" if sub_task != "math" else "Lời giải"}:[/INST] {{ "answer": "{rec[ds_wrapper.target]}", "confident_level": 1}} </s><s>[INST]\n"""

            def format_original_fewshot1(rec):
                return f"""{"Quy luật" if sub_task != "math" else "Bài toán"}: {rec[ds_wrapper.source]}\n{"Kết quả" if sub_task != "math" else "Lời giải"}: {rec[ds_wrapper.target]}\n\n"""

            def format_calib_fewshot(rec):
                return f"""{"Quy luật" if sub_task != "math" else "Bài toán"}: ```\n{rec[ds_wrapper.source]}\n```\n{"Kết quả" if sub_task != "math" else "Lời giải"}:[/INST] {rec[ds_wrapper.target]} </s><s>[INST]\n"""

            selected_sample = list(random.sample(
                list(ds_wrapper.dataset_training), 5))

            original_few_shot = "".join(
                list(
                    map(
                        format_original_fewshot1
                        if self.prompting_strategy == 1
                        else format_original_fewshot0,
                        selected_sample,
                    )
                )
            )
            calib_few_shot = "".join(
                list(
                    map(
                        format_original_fewshot1
                        if self.prompting_strategy == 1
                        else format_calib_fewshot,
                        selected_sample,
                    )
                )
            )

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(few_shot=original_few_shot, rule=rule)
                for rule in batch[ds_wrapper.source]
            ]
            calib_prompts = [
                ds_wrapper.calibration_prompt.format(
                    few_shot=calib_few_shot, rule=rule)
                for rule in batch[ds_wrapper.source]
            ]

            results, logprobs, _ = self.infer_pipeline(
                prompts, return_probs=True)
            calibprob_batch, _ = self.infer_pipeline.compute_logprob_and_length(
                calib_prompts, batch[ds_wrapper.target]
            )
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.target]])
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

    def __translation(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        idx = 0
        original_few_shot = ""
        selected_sample = []
        if self.few_shot_flag:

            def format_original_fewshot0(rec):
                return f"""Khách: "{rec[ds_wrapper.source_language]}"\nBot:[/INST] {{ "translation": "{rec[ds_wrapper.target_language]}" }} </s><s>[INST]\n"""

            def format_original_fewshot1(rec):
                return f"""Đoạn văn: {rec[ds_wrapper.source_language]}\nTrả lời: {rec[ds_wrapper.target_language]}\n\n"""

            selected_sample = list(random.sample(
                list(ds_wrapper.dataset_training), 5))

            original_few_shot = "".join(
                list(
                    map(
                        format_original_fewshot1
                        if self.prompting_strategy == 1
                        else format_original_fewshot0,
                        selected_sample,
                    )
                )
            )

        # Create few-shot strings
        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(
                    few_shot=original_few_shot, document=document)
                for document in batch[ds_wrapper.source_language]
            ]

            results, logprobs, _ = self.infer_pipeline(
                prompts, return_probs=True)
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

        generations = {
            "predictions": predictions,
            "references": references,
            "generation_probs": generation_probs,
            "fewshot": selected_sample,
        }
        saving_fn(generations)

    def run(
        self,
        ds_wrapper,
        ds_loader,
        saving_fn,
        start_idx=0,
        few_shot=False,
        random_mtpc=False,
        prompting_strategy=0,
    ):
        self.prompting_strategy = prompting_strategy
        self.few_shot_flag = few_shot
        self.random_mtpc = random_mtpc
        with torch.no_grad():
            results = self(ds_wrapper, ds_loader, saving_fn, start_idx)
        return results
