import ast
import torch

from tqdm import tqdm
from ..config import GenerationConfig
from ..utils.model import get_model
from ..wrapper import (
    GPTPipeline,
    LLaMaPipeline
)
from ..utils.utils import *
from ..utils.utils_openai import usage_token_from_prompts

class CostEvalPipeline:
    def __init__(self, task, config):
        self.task = task
        extract_task = self.task.split("_")[0]
        self.generation_config = GenerationConfig[extract_task]
        # Load model
        self.model = config.model_name
        # self.model, self.tokenizer = get_model(config=config)
        # self.model.eval()
        # self.infer_pipeline = LLaMaPipeline(
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     generation_config=GenerationConfig[extract_task],
        # )
        self.compute_cost = usage_token_from_prompts
        self.prompting_strategy = 0
        self.few_shot = False
        self.random_mtpc = False
        self.cot = False

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
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        total_cost = 0
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
            num_tokens, cost = self.compute_cost(prompts, encoding_name=self.model, generation_config=self.generation_config)
            total_tokens += num_tokens['total_tokens']
            input_tokens += num_tokens['prompt_tokens']
            output_tokens += num_tokens['completion_tokens']
            total_cost += cost
        return {"total_tokens": total_tokens, "prompt_tokens": input_tokens, "output_tokens": output_tokens}, total_cost

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
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        total_cost = 0
        if self.few_shot:

            def format_original_fewshot0(rec):
                return f"""Câu hỏi: {rec[ds_wrapper.question]}\nCâu trả lời: {{ "answer": "{rec[ds_wrapper.answer]}", "confident_level": 1 }}\n"""

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
                        else format_original_fewshot0,
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
            num_tokens, cost = self.compute_cost(prompts, encoding_name=self.model, generation_config=self.generation_config)
            total_tokens += num_tokens['total_tokens']
            input_tokens += num_tokens['prompt_tokens']
            output_tokens += num_tokens['completion_tokens']
            total_cost += cost
            

        return {"total_tokens": total_tokens, "prompt_tokens": input_tokens, "output_tokens": output_tokens}, total_cost

    def __summarization(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        original_documents = []
        predictions = []
        references = []
        generation_probs = []
        idx = 0
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        total_cost = 0
        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(document=document)
                for document in batch[ds_wrapper.original_text]
            ]
      

            num_tokens, cost = self.compute_cost(prompts, encoding_name=self.model, generation_config=self.generation_config)
            total_tokens += num_tokens['total_tokens']
            input_tokens += num_tokens['prompt_tokens']
            output_tokens += num_tokens['completion_tokens']
            total_cost += cost
            

        return {"total_tokens": total_tokens, "prompt_tokens": input_tokens, "output_tokens": output_tokens}, total_cost


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
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        total_cost = 0
        if self.few_shot:

            def format_original_fewshot0(rec):
                return f"""Khách: "{rec[ds_wrapper.text]}"\nBot: {{ "sentiment": {rec[ds_wrapper.label]}, "confident_level": 1}}\n"""

            def format_original_fewshot1(rec):
                return f"""Đoạn văn: {rec[ds_wrapper.text]}\nQuan điểm: {mapping[rec[ds_wrapper.label]]}\n\n"""

            def format_calib_fewshot(rec):
                return f"""Khách: "{rec[ds_wrapper.text]}"\nBot: {rec[ds_wrapper.label]}\n"""

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
            num_tokens, cost = self.compute_cost(prompts, encoding_name=self.model, generation_config=self.generation_config)
            total_tokens += num_tokens['total_tokens']
            input_tokens += num_tokens['prompt_tokens']
            output_tokens += num_tokens['completion_tokens']
            total_cost += cost

        return {"total_tokens": total_tokens, "prompt_tokens": input_tokens, "output_tokens": output_tokens}, total_cost

    def __multiple_choice_text_classification(
        self, ds_wrapper, ds_loader, saving_fn, start_idx=0
    ):
        sub_task = self.task.split("-")[-1]
        predictions = []
        references = []
        generation_probs = []
        option_probs = []
        idx = 0
        original_few_shot = ""
        calib_few_shot = ""
        selected_sample = []
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        total_cost = 0
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
            num_choice = 7

        elif sub_task == "atis":
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
            num_choice = 17
        else:
            raise ValueError("Invalid sub task")

        if self.few_shot:

            def format_original_fewshot0(rec):
                return f"""Khách: "{rec[ds_wrapper.text]}"\nBot: {{ {"emotion" if sub_task == "vsmec" else "tag"}: {rec[ds_wrapper.label]}, "confident_level": 1}}\n"""

            def format_original_fewshot1(rec):
                return f"""Đoạn văn: {rec[ds_wrapper.text]}\nNhãn: {mapping[rec[ds_wrapper.label]]}\n\n"""

            def format_calib_fewshot(rec):
                return f"""Khách: "{rec[ds_wrapper.text]}"\nBot: {rec[ds_wrapper.label]}\n"""

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
                    cl_samples[random.randint(0, len(cl_samples)-1)])

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

            num_tokens, cost = self.compute_cost(prompts, encoding_name=self.model, generation_config=self.generation_config)
            total_tokens += num_tokens['total_tokens']
            input_tokens += num_tokens['prompt_tokens']
            output_tokens += num_tokens['completion_tokens']
            total_cost += cost
            

        return {"total_tokens": total_tokens, "prompt_tokens": input_tokens, "output_tokens": output_tokens}, total_cost


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
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        total_cost = 0
        if self.few_shot:

            def format_original_fewshot(rec):
                return f"""Khách: "{rec[ds_wrapper.text]}"\nBot: {{ "toxic_level": {rec[ds_wrapper.label]}, "confident_level": 1}}\n"""

            def format_calib_fewshot(rec):
                return f"""Khách: "{rec[ds_wrapper.text]}"\nBot: {rec[ds_wrapper.label]}\n"""

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

            num_tokens, cost = self.compute_cost(prompts, encoding_name=self.model, generation_config=self.generation_config)
            total_tokens += num_tokens['total_tokens']
            input_tokens += num_tokens['prompt_tokens']
            output_tokens += num_tokens['completion_tokens']
            total_cost += cost
            

        return {"total_tokens": total_tokens, "prompt_tokens": input_tokens, "output_tokens": output_tokens}, total_cost


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
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        total_cost = 0
        if self.few_shot:

            def format_original_fewshot(rec):
                return f"""Ngữ cảnh: ''' {rec[ds_wrapper.context]} '''\nCâu hỏi: Hãy lựa chọn đáp án đúng. {rec[ds_wrapper.question]}\n{format_list_ans(rec[ds_wrapper.options])}\n\nCâu trả lời: {{ "choice": "{rec[ds_wrapper.answer]}", "confident_level": 1 }}\n"""

            def format_calib_fewshot(rec):
                return f"""Ngữ cảnh: ''' {rec[ds_wrapper.context]} \nCâu hỏi: Hãy lựa chọn đáp án đúng. {rec[ds_wrapper.question]}\n{format_list_ans(rec[ds_wrapper.options])}\n\nCâu trả lời: {rec[ds_wrapper.answer]}\n"""

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
                    ds_wrapper.prompt.format(
                        few_shot=original_few_shot,
                        context=c,
                        question=q,
                        list_ans=format_list_ans(new_opts),
                    )
                )
               

            num_tokens, cost = self.compute_cost(prompts, encoding_name=self.model, generation_config=self.generation_config)
            total_tokens += num_tokens['total_tokens']
            input_tokens += num_tokens['prompt_tokens']
            output_tokens += num_tokens['completion_tokens']
            total_cost += cost
            

        return {"total_tokens": total_tokens, "prompt_tokens": input_tokens, "output_tokens": output_tokens}, total_cost


    def __language_modelling(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        idx = 0
        original_few_shot = ""
        selected_sample = []
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        total_cost = 0
        if self.few_shot:
            def format_original_fewshot(rec):
                return f"""Khách: "{rec[ds_wrapper.source]}"\nBot: {rec[ds_wrapper.target]}\n"""

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

            num_tokens, cost = self.compute_cost(prompts, encoding_name=self.model, generation_config=self.generation_config)
            total_tokens += num_tokens['total_tokens']
            input_tokens += num_tokens['prompt_tokens']
            output_tokens += num_tokens['completion_tokens']
            total_cost += cost
            

        return {"total_tokens": total_tokens, "prompt_tokens": input_tokens, "output_tokens": output_tokens}, total_cost


    def __information_retrieval(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        # sub_task = self.task.split("_")[1]
        idx = 0
        original_few_shot = ""
        selected_sample = []
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        total_cost = 0
        if self.few_shot:

            def format_original_fewshot(rec):
                return f"""Văn bản: ''' {rec["passage"]} '''\nCâu hỏi: ''' {rec["query"]} '''\n"Văn bản trên có thể hỗ trợ trả lời câu hỏi không?. Đưa ra câu trả lời của bạn dưới dạng JSON với định dạng là ```json {{ \"answer\": ` \"Yes\" or \"No\" `}} ```\nBot: {{ "answer": "{rec["answer"]}" }}\n"""

            def format_calib_fewshot(rec):
                return f"""Văn bản: ''' {rec["passage"]} '''\nCâu hỏi: ''' {rec["query"]} '''\n"Văn bản trên có thể hỗ trợ trả lời câu hỏi không?\nBot: {rec["answer"]}\n"""

            random_sample = list(random.sample(
                list(ds_wrapper.dataset_training), 1))[0]
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
                "query": random_sample[ds_wrapper.query],
                "passage": random_sample["positive"],
                "answer": "Yes",
            }
            second_sample = {
                "query": random_sample[ds_wrapper.query],
                "passage": random_sample["negative"],
                "answer": "No",
            }

            selected_sample = [first_sample, second_sample]

            original_few_shot = "".join(
                list(map(format_original_fewshot, selected_sample))
            )
            calib_few_shot = "".join(
                list(map(format_calib_fewshot, selected_sample)))
        #BATCH_PASSAGE_SIZE = 5
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
        missing = 0
        # Create few-shot strings
        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue
            # print(batch)
            for query_with_a_batch_passages in range(len(batch[ds_wrapper.id])):
                query_id = batch[ds_wrapper.id][query_with_a_batch_passages]
                query = batch[ds_wrapper.query][query_with_a_batch_passages]
                try:
                    ref_passage_id = batch[ds_wrapper.answer][0].tolist()[
                        query_with_a_batch_passages
                    ]
                except:
                    # print(query_id)
                    # print(query)
                    # print(list(batch[ds_wrapper.answer][0]))
                    if len(list(batch[ds_wrapper.answer])) < 1:
                        missing += 1
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
                        ds_wrapper.prompt.format(
                            few_shot=original_few_shot, passage=p, question=query
                        )
                        for p in top30_passages[psg: psg + BATCH_PASSAGE_SIZE]
                    ]
                    num_tokens, cost = self.compute_cost(prompts, encoding_name=self.model, generation_config=self.generation_config)
                    total_tokens += num_tokens['total_tokens']
                    input_tokens += num_tokens['prompt_tokens']
                    output_tokens += num_tokens['completion_tokens']
                    total_cost += cost
            
      
        return {"total_tokens": total_tokens, "prompt_tokens": input_tokens, "output_tokens": output_tokens}, total_cost


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
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        total_cost = 0
        if not self.cot and sub_task == "math":
            target = ds_wrapper.short_target
        else:
            target = ds_wrapper.target

        if self.few_shot:
            def format_original_fewshot0(rec):
                return f"""{"Quy luật" if sub_task != "math" else "Bài toán"}: ```\n{rec[ds_wrapper.source]}\n```\n{"Kết quả" if sub_task != "math" else "Lời giải"}: {{ "answer": "{rec[target]}", "confident_level": 1}}\n"""

            def format_original_fewshot1(rec):
                return f"""{"Quy luật" if sub_task != "math" else "Bài toán"}: {rec[ds_wrapper.source]}\n{"Kết quả" if sub_task != "math" else "Lời giải"}: {rec[target]}\n\n"""

            def format_calib_fewshot(rec):
                return f"""{"Quy luật" if sub_task != "math" else "Bài toán"}: ```\n{rec[ds_wrapper.source]}\n```\n{"Kết quả" if sub_task != "math" else "Lời giải"}: {rec[target]}\n"""

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

            num_tokens, cost = self.compute_cost(prompts, encoding_name=self.model, generation_config=self.generation_config)
            total_tokens += num_tokens['total_tokens']
            input_tokens += num_tokens['prompt_tokens']
            output_tokens += num_tokens['completion_tokens']
            total_cost += cost
            

        return {"total_tokens": total_tokens, "prompt_tokens": input_tokens, "output_tokens": output_tokens}, total_cost


    def __translation(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        idx = 0
        original_few_shot = ""
        selected_sample = []
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        total_cost = 0
        if self.few_shot:

            def format_original_fewshot0(rec):
                return f"""Khách: "{rec[ds_wrapper.source_language]}"\nBot: {{ "translation": "{rec[ds_wrapper.target_language]}" }}\n"""

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

            num_tokens, cost = self.compute_cost(prompts, encoding_name=self.model, generation_config=self.generation_config)
            total_tokens += num_tokens['total_tokens']
            input_tokens += num_tokens['prompt_tokens']
            output_tokens += num_tokens['completion_tokens']
            total_cost += cost
            

        return {"total_tokens": total_tokens, "prompt_tokens": input_tokens, "output_tokens": output_tokens}, total_cost


    def run(
        self,
        ds_wrapper,
        ds_loader,
        saving_fn,
        start_idx=0,
        few_shot=False,
        random_mtpc=False,
        cot=False,
        prompting_strategy=0,
    ):
        self.prompting_strategy = prompting_strategy
        self.few_shot = few_shot
        self.random_mtpc = random_mtpc
        self.cot = cot
        with torch.no_grad():
            results = self(ds_wrapper, ds_loader, saving_fn, start_idx)
            print(" ".join([str(results[0]['total_tokens']), str(results[0]['prompt_tokens']), str(results[0]['output_tokens']), str(results[1])]))
        return results
