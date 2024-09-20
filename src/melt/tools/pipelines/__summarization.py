"__summarization"
import random
from tqdm import tqdm
from melt.tools.utils.utils import format_fewshot

def __summarization(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
    original_documents = []
    predictions = []
    original_few_shot = []
    selected_sample = []
    references = []
    generation_probs = []
    if self.continue_infer_data is not None:
        original_documents.extend(
            self.continue_infer_data["original_documents"]
        )
        predictions.extend(self.continue_infer_data["predictions"])
        references.extend(self.continue_infer_data["references"])
        generation_probs.extend(
            self.continue_infer_data["generation_probs"]
        )
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
                {
                    "role": "system",
                    "content": ds_wrapper.prompt["system_prompt"],
                },
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
        original_documents.extend(list(batch[ds_wrapper.dataset_info.source]))

        results, logprobs, _ = self.infer_pipeline(
            prompts, return_probs=True
        )
        predictions.extend(results)
        references.extend(list(batch[ds_wrapper.dataset_info.target]))
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
