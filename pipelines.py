from tqdm import tqdm
from generation_config import GenerationConfig


class InferPipeline:
    def __init__(self, model, tokenizer, generation_config):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

    def __call__(self, prompts, return_probs=False):
        inputs = self.tokenizer(
            prompts, return_tensors="pt").to(self.model.device)
        generate_dict = self.model.generate(
            inputs.input_ids,
            output_scores=True,
            return_dict_in_generate=True,
            **self.generation_config
        )

        num_generated_tokens = len(generate_dict.scores)
        generated_tokens = generate_dict.sequences[:, -num_generated_tokens:]

        generations = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True)

        if return_probs:
            generations_probs = self.model.compute_transition_scores(
                sequences=generated_tokens,
                scores=generate_dict.scores,
                normalize_logits=True
            )

            return generations, generations_probs
        else:
            return generations


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
        elif task == "text-generation":
            return self.__text_generation(ds_wrapper, ds_loader, saving_fn, start_idx)
        else:
            raise NotImplementedError

    def __question_answering(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
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

            results = self.infer_pipeline(prompts, return_probs=False)
            predictions.extend(results)
            references.extend([x[0] for x in batch[ds_wrapper.answer]["text"]])

            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {"predictions": predictions,
                               "references": references}
                saving_fn(generations)

        generations = {"predictions": predictions, "references": references}
        saving_fn(generations)

    def __summarization(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
        idx = 0

        for batch in tqdm(ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            prompts = [
                ds_wrapper.prompt.format(document=document)
                for document in batch[ds_wrapper.original_text]
            ]

            results = self.infer_pipeline(prompts, return_probs=False)
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.summarized_text]])

            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {"predictions": predictions,
                               "references": references}
                saving_fn(generations)

        generations = {"predictions": predictions, "references": references}
        saving_fn(generations)

    def __multiple_choice(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        pass

    def __translation(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        predictions = []
        references = []
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

            results = self.infer_pipeline(prompts, return_probs=False)
            predictions.extend(results)
            references.extend([x for x in batch[ds_wrapper.target_language]])

            idx += 1
            if idx % 100 == 0:
                print(f"Saving results of {idx} batches")
                generations = {"predictions": predictions,
                               "references": references}
                saving_fn(generations)

        generations = {"predictions": predictions, "references": references}
        saving_fn(generations)

    def __text_generation(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        pass

    def run(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
        results = self(ds_wrapper, ds_loader, saving_fn, start_idx)
        return results
