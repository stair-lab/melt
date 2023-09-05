import transformers

from tqdm import tqdm
from generation_config import GenerationConfig


class EvalPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = GenerationConfig()

    def __call__(self, ds_wrapper, ds_loader, saving_fn=None):
        if self.task == "question-answering":
            return self.__question_answering(ds_wrapper, ds_loader, saving_fn)
        elif self.task == "summarization":
            return self.__summarization(ds_wrapper, ds_loader, saving_fn)
        elif self.task == "translation":
            return self.__translation(ds_wrapper, ds_loader, saving_fn)
        elif self.task == "text-generation":
            return self.__text_generation(ds_wrapper, ds_loader, saving_fn)
        else:
            raise NotImplementedError

    def __question_answering(self, ds_wrapper, ds_loader, saving_fn=None):
        pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            task="text-generation",
            **self.generation_config.question_answering,
        )

        if "gpt2" in self.model.name_or_path:
            context_truncate_length = 700
            question_truncate_length = 200
        else:
            context_truncate_length = None
            question_truncate_length = None

        predictions = []
        references = []
        i = 0
        for batch in tqdm(ds_loader):
            prompts = [
                ds_wrapper.prompt.format(
                    context=c[:context_truncate_length],
                    question=q[:question_truncate_length],
                )
                for c, q in zip(batch[ds_wrapper.context], batch[ds_wrapper.question])
            ]

            results = pipeline(prompts, batch_size=ds_loader.batch_size)
            predictions.extend([x[0]["generated_text"] for x in results])
            references.extend([x[0] for x in batch[ds_wrapper.answer]["text"]])

            i += 1
            if i % 100 == 0:
                print(f"Saving results of {i} batches")
                generations = {"predictions": predictions,
                               "references": references}
                saving_fn(generations)

        generations = {"predictions": predictions, "references": references}
        saving_fn(generations)

        # metric = load("exact_match")
        # results = metric.compute(predictions=predictions,
        #                             references=references,
        #                             ignore_case =True,
        #                             ignore_punctuation=True
        # )

    def __summarization(self, ds_wrapper, ds_loader, saving_fn=None):
        pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            task="text-generation",
            **self.generation_config.summarization,
        )
        if "gpt2" in self.model.name_or_path:
            truncate_length = 700
        else:
            truncate_length = None
        predictions = []
        references = []
        i = 0
        for batch in tqdm(ds_loader):
            prompts = [
                ds_wrapper.prompt.format(document=document[:truncate_length])
                for document in batch[ds_wrapper.original_text]
            ]

            results = pipeline(prompts, batch_size=ds_loader.batch_size)
            predictions.extend(
                [x[0]["generated_text"].split("\n\n")[0] for x in results]
            )
            references.extend([x for x in batch[ds_wrapper.summarized_text]])

            i += 1
            if i % 100 == 0:
                print(f"Saving results of {i} batches")
                generations = {"predictions": predictions,
                               "references": references}
                saving_fn(generations)

        generations = {"predictions": predictions, "references": references}
        saving_fn(generations)

        # metrics = load('rouge')
        # results = metrics.compute(predictions=predictions,
        #                           references=references
        # )

    def __translation(self, ds_wrapper, ds_loader, saving_fn=None):
        pass

    def __text_generation(self, ds_wrapper, ds_loader, saving_fn=None):
        pass

    def run(self, ds_wrapper, ds_loader, saving_fn=None):
        results = self(ds_wrapper, ds_loader, saving_fn)
        return results
