"__translation"
from tqdm import tqdm

def __translation(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
    # Group related variables into a dictionary
    results_data = {
        "predictions": [],
        "references": [],
        "generation_probs": [],
    }
    # Helper function to save generations and compute results
    def save_results(idx, generations):
        print(f"Saving results of {idx} batches")
        saving_fn(generations)
        mean_result = self.metric_pipeline.run_mean(
            generations,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        )
        print(f"Results of {idx} batches: ", mean_result)

    idx = 0
    original_few_shot = []

    if self.continue_infer_data is not None:
        results_data["predictions"].extend(self.continue_infer_data["predictions"])
        results_data["references"].extend(self.continue_infer_data["references"])
        results_data["generation_probs"].extend(self.continue_infer_data["generation_probs"])

    if self.few_shot:
        # Extract few-shot data into a separate function
        _, original_few_shot = self.get_few_shot(ds_wrapper)

    # Create few-shot strings and process batches
    for batch in tqdm(ds_loader):
        if idx < start_idx:
            idx += 1
            continue

        # Inline prompts construction
        prompts = [
            [
                {"role": "system", "content": ds_wrapper.prompt["system_prompt"]},
                *original_few_shot,
                {"role": "user", "content": ds_wrapper.prompt["prompt"].format(document)},
            ]
            for document in batch[ds_wrapper.dataset_info.source]
        ]

        results, logprobs, _ = self.infer_pipeline(prompts, return_probs=True)
        results_data["predictions"].extend(results)
        results_data["references"].extend(list(
            batch[ds_wrapper.dataset_info.target]))# Fixed unnecessary comprehension
        results_data["generation_probs"].extend(logprobs)
        idx += 1
        if idx % 100 == 0:
            save_results(idx, results_data)
    # Save generations and compute final results
    final_result = {
        "mean": self.metric_pipeline.run_mean(
            results_data,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        ),
        "std": self.metric_pipeline.run_std(
            results_data,
            self.task_name,
            ds_wrapper.prompt["answer_key"],
            ds_wrapper.dataset_info.label,
            self.config,
        ),
    }

    saving_fn(results_data, final_result)
