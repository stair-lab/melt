"math"
import random
from tqdm import tqdm
from melt.tools.utils.utils import format_fewshot
def __math(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
    predictions = []
    references = []
    generation_probs = []
    calib_probs = []
    math_problem_type = []
    idx = 0
    original_few_shot = []
    calib_few_shot = []
    selected_sample = []

    if self.continue_infer_data is not None:
        predictions.extend(self.continue_infer_data["predictions"])
        references.extend(self.continue_infer_data["references"])
        generation_probs.extend(self.continue_infer_data["generation_probs"])
        calib_probs.extend(self.continue_infer_data["calibration_probs"])
        math_problem_type.extend(self.continue_infer_data.get("math_problem_type", []))

    if self.few_shot:

        def preprocessing_a_record(rec):
            return [
                rf"{rec[ds_wrapper.dataset_info.query]}",
                rf"{rec[ds_wrapper.dataset_info.answer]}",
            ]

        selected_sample = [
            preprocessing_a_record(s)
            for s in list(
                random.sample(
                    list(ds_wrapper.dataset_training), self.config.num_fs
                )
            )
        ]
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
                        rf"{rule}"
                    ),
                },
            ]
            for rule in batch[ds_wrapper.dataset_info.query]
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
                    "content": ds_wrapper.calibration_prompt["prompt"].format(rf"{rule}"),
                },
            ]
            for rule in batch[ds_wrapper.dataset_info.query]
        ]

        results, logprobs, _ = self.infer_pipeline(
            prompts, return_probs=True
        )
        calibprob_batch, _ = (
            self.infer_pipeline.compute_logprob_and_length(
                calib_prompts, batch[ds_wrapper.dataset_info.answer]
            )
        )
        predictions.extend(results)
        references.extend(list(batch[ds_wrapper.dataset_info.answer]))
        generation_probs.extend(logprobs)
        calib_probs.extend(calibprob_batch)
        math_problem_type.extend(list(batch[ds_wrapper.dataset_info.type_id]))
        idx += 1
        if idx % 100 == 0:
            print(f"Saving results of {idx} batches")
            generations = {
                "predictions": predictions,
                "references": references,
                "generation_probs": generation_probs,
                "calibration_probs": calib_probs,
                "fewshot": selected_sample,
                "math_problem_type": math_problem_type,
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
        "calibration_probs": calib_probs,
        "fewshot": selected_sample,
        "math_problem_type": math_problem_type,
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