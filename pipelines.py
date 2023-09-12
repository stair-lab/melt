import torch

from tqdm import tqdm
from generation_config import GenerationConfig
import ast

class InferPipeline:
    def __init__(self, model, tokenizer, generation_config):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

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
                **self.generation_config
            )

            num_generated_token = len(generate_dict.scores)
            num_generated_tokens.append(num_generated_token)
            generated_tokens = generate_dict.sequences[:, -
                                                       num_generated_token:]

            generation = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True)
            generations.extend(generation)

            if return_probs:
                generation_probs = self.model.compute_transition_scores(
                    sequences=generated_tokens,
                    scores=generate_dict.scores,
                    normalize_logits=True
                )
                generations_probs.extend(generation_probs.cpu().numpy())

        return generations, generations_probs, num_generated_tokens

    def compute_logprob_and_length(self, prompts, completions):
        completions_num_tokens = []
        completions_logprobs = []

        for prompt, completion in zip(prompts, completions):
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt").to(
                self.model.device)  # <s> [tokens]
            # Actual number of tokens in completion (without `<s>`)
            prompt_num_tokens = prompt_tokens.input_ids.shape[1] - 1

            completion_tokens = self.tokenizer(
                completion, return_tensors="pt").to(self.model.device)  # <s> [tokens]
            # Actual number of tokens in completion (without `<s>`)
            completion_num_tokens = completion_tokens.input_ids.shape[1] - 1
            completions_num_tokens.append(completion_num_tokens)

            inputs = torch.concatenate(
                (prompt_tokens.input_ids, completion_tokens.input_ids[:, 1:]), dim=-1)
            outputs = self.model(inputs)  # [input_tokens] [next_token]

            logits = outputs.logits[:,
                                    prompt_num_tokens:prompt_num_tokens+completion_num_tokens]
            logprobs = logits.log_softmax(dim=-1)
            # >>> batch_size, sequence_length, vocab_size

            logprobs = logprobs.gather(
                dim=-1, index=completion_tokens.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            # >>> batch_size, sequence_length
            completions_logprobs.append(logprobs.cpu().numpy())

        return completions_logprobs, completions_num_tokens


class EvalPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.infer_pipeline = InferPipeline(
            model=model,
            tokenizer=tokenizer,
            generation_config=GenerationConfig[task]
        )

    def __call__(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        task = self.task.split("_")[0]

        if task == "question-answering":
            return self.__question_answering(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif task == "summarization":
            return self.__summarization(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif task == "translation":
            return self.__translation(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif task == "language-modelling":
            return self.__language_modelling(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif task == "text-classification":
            return self.__multiple_choice_text_classification(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif task == "sentiment-analysis":
            return self.__multiple_choice_sentiment(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif task == "toxic-detection":
            return self.__multiple_choice_toxicity(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif task == "knowledge":
            subtask = self.task.split("_")[1]
            if subtask == "mtpchoice":
                return self.__multiple_choice(ds_wrapper, ds_loader, saving_fn, start_idx)
            elif subtask == "openended":
                return self.__question_answering_without_context(ds_wrapper, ds_loader, saving_fn, start_idx)
            else:
                raise NotImplementedError
        elif task == "information-retrieval":
            return self.__information_retrieval(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif task == "reasoning":
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
            generation_probs.extend([x.mean() for x in logprobs])

            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {"predictions": predictions,
                               "references": references,
                               "generation_probs": generation_probs}
                saving_fn(generations)

        generations = {"predictions": predictions,
                       "references": references,
                       "generation_probs": generation_probs}
        saving_fn(generations)
    
    def __question_answering_without_context(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
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
                    question=q,
                )
                for q in batch[ds_wrapper.question]
            ]

            results, logprobs, _ = self.infer_pipeline(
                prompts, return_probs=True)
            predictions.extend(results)
            references.extend([x[0] for x in batch[ds_wrapper.answer]])
            generation_probs.extend([x.mean() for x in logprobs])

            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {"predictions": predictions,
                               "references": references,
                               "generation_probs": generation_probs}
                saving_fn(generations)

        generations = {"predictions": predictions,
                       "references": references,
                       "generation_probs": generation_probs}
        saving_fn(generations)

    def __summarization(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
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

            results, logprobs, _ = self.infer_pipeline(
                prompts, return_probs=True)
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.summarized_text]])
            generation_probs.extend([x.mean() for x in logprobs])

            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {"predictions": predictions,
                               "references": references,
                               "generation_probs": generation_probs}
                saving_fn(generations)

        generations = {"predictions": predictions,
                       "references": references,
                       "generation_probs": generation_probs}
        saving_fn(generations)
    
    def __multiple_choice_sentiment(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        option_probs = []
        idx = 0

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(
                    context=c,
                )
                for c in batch[ds_wrapper.text]
            ]

            results, logprobs, _  = self.infer_pipeline(prompts, return_probs=True)
            num_choice = 3
            
            option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                prompts*num_choice, [f"{{ \"sentiment\": \"{choice}\", \"confident_level\": 1 }}" for choice in range(num_choice) for dup in range(len(prompts))])
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.label]])
            generation_probs.extend([x.mean() for x in logprobs])
            option_probs.extend([[option_logprobs[idx+opt*len(prompts)] for opt in range(num_choice)] for idx in range(len(prompts))])
            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {"predictions": predictions,
                               "references": references,
                               "generation_probs": generation_probs,
                               "option_probs": option_probs}
            
                saving_fn(generations)

        generations = {"predictions": predictions,
                       "references": references,
                       "generation_probs": generation_probs,
                        "option_probs": option_probs }
        saving_fn(generations)
    
    def __multiple_choice_text_classification(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        sub_task = self.task.split('_')[1]
        predictions = []
        references = []
        generation_probs = []
        option_probs = []
        idx = 0

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(
                    context=c,
                )
                for c in batch[ds_wrapper.text]
            ]

            results, logprobs, _  = self.infer_pipeline(prompts, return_probs=True)
            if sub_task == "vsmec":
                num_choice = 7
                option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                    prompts*num_choice, [f"{{ \"emotion\": \"{choice}\", \"confident_level\": 1 }}" for choice in range(num_choice) for _ in range(len(prompts))])
            
            else:
                num_choice = 17
                option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                    prompts*num_choice, [f"{{ \"tag\": \"{choice}\", \"confident_level\": 1 }}" for choice in range(num_choice) for _ in range(len(prompts))])
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.label]])
            generation_probs.extend([x.mean() for x in logprobs])
            option_probs.extend([[option_logprobs[idx+opt*len(prompts)] for opt in range(num_choice)] for idx in range(len(prompts))])
            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {"predictions": predictions,
                               "references": references,
                               "generation_probs": generation_probs,
                               "option_probs": option_probs}
            
                saving_fn(generations)

        generations = {"predictions": predictions,
                       "references": references,
                       "generation_probs": generation_probs,
                        "option_probs": option_probs }
        saving_fn(generations)
    
    def __multiple_choice_toxicity(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        sub_task = self.task.split('_')[1]
        predictions = []
        references = []
        generation_probs = []
        option_probs = []
        idx = 0

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(
                    context=c,
                )
                for c in batch[ds_wrapper.text]
            ]

            results, logprobs, _  = self.infer_pipeline(prompts, return_probs=True)
            num_choice = 2 if sub_task == "ViCTSD" else 3
            
            option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                prompts*num_choice, [f"{{ \"toxicity_level\": \"{choice}\", \"confident_level\": 1 }}" for choice in range(num_choice) for _ in range(len(prompts))])
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.label]])
            generation_probs.extend([x.mean() for x in logprobs])
            option_probs.extend([[option_logprobs[idx+opt*len(prompts)] for opt in range(num_choice)] for idx in range(len(prompts))])
            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {"predictions": predictions,
                               "references": references,
                               "generation_probs": generation_probs,
                               "option_probs": option_probs}
            
                saving_fn(generations)

        generations = {"predictions": predictions,
                       "references": references,
                       "generation_probs": generation_probs,
                        "option_probs": option_probs }
        saving_fn(generations)
    
    def __multiple_choice(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        format_list_ans = lambda ans_list:'\n'.join(list(map(lambda ans: f"{chr(ans[0]+65)}: ''' {ans[1]} '''", enumerate(ans_list))))
        predictions = []
        references = []
        generation_probs = []
        option_probs = []
        idx = 0

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(
                    context=c,
                    question=q,
                    list_ans=format_list_ans(ast.literal_eval(opts)),
                )
                for c, q, opts in zip(batch[ds_wrapper.context], batch[ds_wrapper.question], batch[ds_wrapper.options])
            ]

            results, logprobs, _  = self.infer_pipeline(prompts, return_probs=True)
            option_logprobs, _ = self.infer_pipeline.compute_logprob_and_length(
                prompts*4, [f"{{ \"choice\": \"{chr(choice+65)}\", \"confident_level\": 1 }}" for choice in range(4) for dup in range(len(prompts))])
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.answer]])
            generation_probs.extend([x.mean() for x in logprobs])
            option_probs.extend([[option_logprobs[idx+opt*len(prompts)] for opt in range(4)] for idx in range(len(prompts))])
            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {"predictions": predictions,
                               "references": references,
                               "generation_probs": generation_probs,
                               "option_probs": option_probs}
            
                saving_fn(generations)

        generations = {"predictions": predictions,
                       "references": references,
                       "generation_probs": generation_probs,
                        "option_probs": option_probs }
        saving_fn(generations)

    def __language_modelling(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        idx = 0

        # Create few-shot strings
        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(context=c)
                for c in batch[ds_wrapper.context]
            ]

            results, logprobs, _ = self.infer_pipeline(
                prompts, return_probs=True)
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.target]])
            generation_probs.extend([x.mean() for x in logprobs])

            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {"predictions": predictions,
                               "references": references,
                               "generation_probs": generation_probs}
                saving_fn(generations)

        generations = {"predictions": predictions,
                       "references": references,
                       "generation_probs": generation_probs}
        saving_fn(generations)

    def __information_retrieval(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        pass

    def __reasoning(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        math_problem_type = []
        sub_task = self.task.split('_')[1]
        idx = 0

        # Create few-shot strings
        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(rule=rule)
                for rule in batch[ds_wrapper.source]
            ]

            results, logprobs, _ = self.infer_pipeline(
                prompts, return_probs=True)
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.target]])
            generation_probs.extend([x.mean() for x in logprobs])
            if sub_task == "math":
                math_problem_type.extend([x for x in batch[ds_wrapper.type]])
            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {"predictions": predictions,
                               "references": references,
                               "generation_probs": generation_probs}
                if sub_task == "math":
                    generations["math_problem_type"] = math_problem_type
                saving_fn(generations)

        generations = {"predictions": predictions,
                       "references": references,
                       "generation_probs": generation_probs}
        if sub_task == "math":
            generations["math_problem_type"] = math_problem_type
        saving_fn(generations)

    def __translation(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        generation_probs = []
        idx = 0

        # Create few-shot strings
        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(document=document)
                for document in batch[ds_wrapper.source_language]
            ]

            results, logprobs, _ = self.infer_pipeline(
                prompts, return_probs=True)
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.target_language]])
            generation_probs.extend([x.mean() for x in logprobs])

            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {"predictions": predictions,
                               "references": references,
                               "generation_probs": generation_probs}
                saving_fn(generations)

        generations = {"predictions": predictions,
                       "references": references,
                       "generation_probs": generation_probs}
        saving_fn(generations)

    def run(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        with torch.no_grad():
            results = self(ds_wrapper, ds_loader, saving_fn, start_idx)
        return results
