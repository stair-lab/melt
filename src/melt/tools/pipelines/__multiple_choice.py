"multiple choice"
import ast
import random
from tqdm import tqdm
from melt.tools.utils.utils import format_fewshot
def __multiple_choice(self, ds_wrapper, ds_loader, saving_fn, start_idx=0):
    def format_list_ans(ans_list):
        return "\n".join(
            list(
                map(
                    lambda ans:
                    f"{ds_wrapper.dataset_info.label[ans[0]]}: \
                    ''' {ans[1]} '''",
                    enumerate(ans_list),
                )
            )
        )

    predictions = []
    references = []
    generation_probs = []
    option_probs = []
    idx = 0
    original_few_shot = []
    calib_few_shot = []
    option_order_all = []
    selected_sample = []
    # alphabet2idx = {chr(i + 65): i for i in range(26)}
    num_choice = len(ds_wrapper.dataset_info.label)
    if self.continue_infer_data is not None:
        predictions.extend(self.continue_infer_data["predictions"])
        references.extend(self.continue_infer_data["references"])
        generation_probs.extend(
            self.continue_infer_data["generation_probs"]
        )
        option_probs.extend(self.continue_infer_data["option_probs"])
        option_order_all.extend(self.continue_infer_data["option_orders"])

    if self.few_shot:

        def preprocessing_a_record(rec):
            return [
                rec[ds_wrapper.dataset_info.context],
                rec[ds_wrapper.dataset_info.query],
                format_list_ans(
                    ast.literal_eval(rec[ds_wrapper.dataset_info.options])
                ),
                rec[ds_wrapper.dataset_info.answer],
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
        calib_few_shot = format_fewshot(
            selected_sample,
            query_format=ds_wrapper.calibration_prompt["prompt"],
            answer_format=ds_wrapper.prompt["answer_format"],
        )
    for batch in tqdm(ds_loader):
        if idx < start_idx:
            idx += 1
            continue
        prompts = []
        calib_prompts = []
        remap_order_batch = []
        for cq in zip(
            batch[ds_wrapper.dataset_info.context],
            batch[ds_wrapper.dataset_info.query],
            batch[ds_wrapper.dataset_info.options],
        ):
            c = cq[0]
            q = cq[1]
            opts = ast.literal_eval(cq[2])
            order_shuffle = list(range(len(opts)))
            if ds_wrapper.dataset_info.random:
                random.shuffle(order_shuffle)
            remap_order_batch.append(order_shuffle)
            new_opts = [opts[i] for i in order_shuffle]
            prompts.append(
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
                            q,
                            format_list_ans(new_opts),
                        ),
                    },
                ]
            )
            calib_prompts.append(
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
                            q,
                            format_list_ans(new_opts),
                        ),
                    },
                ]
            )

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
        opt_calib_out = [
            [
                option_logprobs[i + opt * len(prompts)]
                for opt in range(num_choice)
            ]
            for i in range(len(prompts))
        ]

        # Reshuffle answer of calib
        option_order_all.extend(remap_order_batch)
        predictions.extend(results)
        # In case order of options is changed
        # Map the reference to the new order
        references.extend(
            [
                ds_wrapper.dataset_info.label[
                    remap.index(ds_wrapper.dataset_info.label.index(x))
                ]
                for x, remap in zip(
                    batch[ds_wrapper.dataset_info.answer],
                    remap_order_batch,
                )
            ]
        )

        generation_probs.extend(logprobs)
        option_probs.extend(opt_calib_out)
        idx += 1
        if idx % 100 == 0:
            print(f"Saving results of {idx} batches")
            generations = {
                "predictions": predictions,
                "references": references,  # new order
                "generation_probs": generation_probs,
                "option_probs": option_probs,  # new order
                "option_orders": option_order_all,
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
        "option_orders": option_order_all,
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
