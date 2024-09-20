"information retrieval"
import random
from tqdm import tqdm
from melt.tools.utils.utils import column, format_fewshot

def __information_retrieval(
    self, ds_wrapper, ds_loader, saving_fn, start_idx=0
):
    predictions = []
    idx = 0
    original_few_shot = []
    calib_few_shot = []
    selected_sample = []
    if self.few_shot:
        def preprocessing_a_record(rec):
            return [
                rec[ds_wrapper.dataset_info.passages],
                rec[ds_wrapper.dataset_info.query],
                rec[ds_wrapper.dataset_info.answer],
            ]

        random_sample = list(
            random.sample(list(ds_wrapper.dataset_training), 1)
        )[0]
        first_sample = {
            "passages": random_sample["positive"],
            "query": random_sample[ds_wrapper.dataset_info.query],
            "references": ds_wrapper.dataset_info.label[0],
        }
        second_sample = {
            "passages": random_sample["negative"],
            "query": random_sample[ds_wrapper.dataset_info.query],
            "references": ds_wrapper.dataset_info.label[1],
        }

        selected_sample = [
            preprocessing_a_record(s)
            for s in [first_sample, second_sample]
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

    batch_passage_size = 10
    # Create few-shot strings
    for batch in tqdm(ds_loader):
        if idx < start_idx:
            idx += 1
            continue
        for query_with_a_batch_passages in range(
            len(batch[ds_wrapper.dataset_info.type_id])
        ):
            query_id = batch[ds_wrapper.dataset_info.type_id][
                query_with_a_batch_passages
            ]
            query = batch[ds_wrapper.dataset_info.query][
                query_with_a_batch_passages
            ]
            try:
                ref_passage_id = batch[ds_wrapper.dataset_info.answer][0][
                    query_with_a_batch_passages
                ]
            except IndexError:
                if len(list(batch[ds_wrapper.dataset_info.answer])) < 1:
                    continue
                ref_passage_id = list(
                    batch[ds_wrapper.dataset_info.answer][0]
                )[query_with_a_batch_passages]
            batch_passages = batch[ds_wrapper.dataset_info.passages]

            top30_passage_ids = column(
                batch_passages["id"], query_with_a_batch_passages
            )
            top30_passages = column(
                batch_passages["passage"], query_with_a_batch_passages
            )
            for psg in range(
                0, len(top30_passage_ids), batch_passage_size
            ):
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
                                p,
                                query,
                            ),
                        },
                    ]
                    for p in top30_passages[psg:psg + batch_passage_size]
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
                                p,
                                query,
                            ),
                        },
                    ]
                    for p in top30_passages[psg:psg + batch_passage_size]
                ]
                results, logprobs, _ = self.infer_pipeline(
                    prompts, return_probs=True
                )

                option_logprobs, _ = (
                    self.infer_pipeline.compute_logprob_and_length(
                        calib_prompts * len(ds_wrapper.dataset_info.label),
                        [
                            choice
                            for choice in ds_wrapper.dataset_info.label
                            for _ in range(len(prompts))
                        ],
                    )
                )
                # Use a separate function to avoid cell-var-from-loop warnings
                def create_prompt_dict(data):
                    return {
                        "query_id": (
                            data['query_id'].item()
                            if not isinstance(data['query_id'], str)
                            else data['query_id']
                        ),
                        "query": data['query'],
                        "passage_id": (
                            data['passage_id'].item() if not isinstance(
                                data['passage_id'], str) else data['passage_id']
                        ),
                        "passage": data['passage'],
                        "label": int(
                            data['passage_id'].item() == data['ref_passage_id']
                            if not isinstance(data['passage_id'], str)
                            else data['passage_id'] == data['ref_passage_id']
                        ),
                        "prediction": data['prediction'],
                        "generation_probs": data['generation_probs'],
                        "calib_probs": [
                            data['option_logprobs'][data['q'] + opt * len(data['prompts'])]
                            for opt in range(
                                len(ds_wrapper.dataset_info.label)
                            )
                        ],
                    }
                save_each_prompt = [
                    create_prompt_dict({
                        'prediction': x,
                        'generation_probs': y,
                        'passage_id': z,
                        'passage': t,
                        'q': q,
                        'query_id': query_id,
                        'query': query,
                        'ref_passage_id': ref_passage_id,
                        'option_logprobs': option_logprobs,
                        'prompts': prompts
                    })
                    for x, y, z, t, q in zip(
                        results,
                        logprobs,
                        top30_passage_ids[psg:psg + batch_passage_size],
                        top30_passages[psg:psg + batch_passage_size],
                        range(len(prompts))
                    )
                ]
                predictions.extend(save_each_prompt)

        idx += 1

        if idx % 100 == 0:
            print(f"Saving results of {idx} batches")
            generations = {
                "fewshot": selected_sample,
                "predictions": predictions,
            }
            saving_fn(generations)
            mean_result = self.metric_pipeline.run_mean(
                generations,
                self.task_name,
                ds_wrapper.prompt["answer_key"],
                ds_wrapper.dataset_info.label,
                self.config,
                ref_dataset=ds_wrapper.dataset_testing,
            )
            print(f"Results of {idx} batches: ", mean_result)

    generations = {"fewshot": selected_sample, "predictions": predictions}
    mean_result = self.metric_pipeline.run_mean(
        generations,
        self.task_name,
        ds_wrapper.prompt["answer_key"],
        ds_wrapper.dataset_info.label,
        self.config,
        ref_dataset=ds_wrapper.dataset_testing,
    )
    std_result = self.metric_pipeline.run_std(
        generations,
        self.task_name,
        ds_wrapper.prompt["answer_key"],
        ds_wrapper.dataset_info.label,
        self.config,
        ref_dataset=ds_wrapper.dataset_testing,
    )
    final_result = {"mean": mean_result, "std": std_result}
    saving_fn(generations, final_result)
