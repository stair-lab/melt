"multiple choice sentiment"
import random
from tqdm import tqdm
from melt.tools.utils.utils import format_fewshot, unique

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
        generation_probs.extend(
            self.continue_infer_data["generation_probs"]
        )
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
                lambda r, class_label=cl: r[ds_wrapper.dataset_info.answer] == class_label
            )
            selected_sample.append(
                preprocessing_a_record(
                    cl_samples[random.randint(0, len(cl_samples) - 1)]
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
                {
                    "role": "system",
                    "content": ds_wrapper.prompt["system_prompt"],
                },
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
                        c,
                    ),
                },
            ]
            for c in batch[ds_wrapper.dataset_info.query]
        ]
        results, logprobs, _ = self.infer_pipeline(
            prompts, return_probs=True
        )

        option_logprobs, _ = (
            self.infer_pipeline.compute_logprob_and_length(
                calib_prompts * num_choice,
                [
                    ds_wrapper.dataset_info.label[choice]
                    for choice in range(num_choice)
                    for _ in range(len(prompts))
                ],
            )
        )
        predictions.extend(results)
        references.extend(
            [x.item() for x in batch[ds_wrapper.dataset_info.answer]]
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
