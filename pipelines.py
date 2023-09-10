import torch

from tqdm import tqdm
from generation_config import GenerationConfig


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
            return self.__multiple_choice(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif task == "sentiment-analysis":
            return self.__multiple_choice(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif task == "toxic-detection":
            return self.__multiple_choice(ds_wrapper, ds_loader, saving_fn, start_idx)
        elif task == "knowledge":
            subtask = self.task.split("_")[1]
            if subtask == "mtpchoice":
                return self.__multiple_choice(ds_wrapper, ds_loader, saving_fn, start_idx)
            elif subtask == "openended":
                return self.__question_answering(ds_wrapper, ds_loader, saving_fn, start_idx)
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

    def __multiple_choice(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        pass

    def __language_modelling(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        pass

    def __information_retrieval(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        pass

    def __reasoning(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        pass

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
