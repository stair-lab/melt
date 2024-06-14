import os
from .tools.data import DatasetWrapper
from .tools.pipelines import EvalPipeline
from .tools.utils.utils import save_to_json, set_seed, read_json
from torch.utils.data import DataLoader


def generation(script_args):
    set_seed(script_args.seed)

    # Save results
    if not os.path.exists(script_args.output_dir):
        os.makedirs(script_args.output_dir)
    if not os.path.exists(script_args.output_eval_dir):
        os.makedirs(script_args.output_eval_dir)

    ds_exact_name = (
        script_args.dataset_name.split("/")[-1]
        + "_"
        + script_args.model_name.split("/")[-1]
        + f"_pt{script_args.prompting_strategy}"
        + ("_fewshot" if script_args.fewshot_prompting else "")
        + ("_randchoice" if script_args.random_mtpc else "")
        + ("_cot" if script_args.cot else "")
        + f"_seed{script_args.seed}"
    )

    json_file = os.path.join(
        script_args.output_dir, f"generations_{ds_exact_name}.json"
    )
    metric_file = os.path.join(script_args.output_eval_dir, f"metrics_{ds_exact_name}.json")

    if script_args.continue_infer:
        if os.path.exists(json_file):
            # df1, fewshots = read_json(json_file)
            continue_results, current_batch_idx = read_json(
                json_file, script_args.per_device_eval_batch_size
            )
            start_idx = current_batch_idx
        else:
            start_idx = 0
            continue_results = None
    else:
        start_idx = 0
        continue_results = None
    fewshots = None

    # Load dataset (you can process it here)
    dataset_wrapper = DatasetWrapper(
        dataset_name=script_args.dataset_name,
        config_dir = script_args.config_dir,
        prompting_strategy=script_args.prompting_strategy,
        fewshots=fewshots,
    )
    if script_args.smoke_test:
        n_examples = 8
        dataset_wrapper.dataset_testing = dataset_wrapper.dataset_testing.select(
            range(n_examples)
        )

    dataset_loader = DataLoader(
        dataset_wrapper.get_dataset_testing(),
        batch_size=script_args.per_device_eval_batch_size,
        shuffle=False,
    )

    # Initialize pipeline
    eval_pipeline = EvalPipeline(task=dataset_wrapper.task, config=script_args)

    # Evaluate
    def save_results(generations, metrics=None):
        # if script_args.continue_infer and os.path.exists(json_file):
        # df2 = pd.DataFrame(generations)
        # df3 = df1.append(df2, ignore_index=True)
        #    generations = df3.to_dict(orient="list")

        save_to_json(generations, json_file)
        if metrics is not None:
            save_to_json(metrics, metric_file)

    eval_pipeline.run(
        ds_wrapper=dataset_wrapper,
        ds_loader=dataset_loader,
        generation_results_file=ds_exact_name,
        saving_fn=save_results,
        start_idx=start_idx,
        few_shot=script_args.fewshot_prompting,  # few-shot prompting
        random_mtpc=script_args.random_mtpc,  # random multiple choice
        cot=script_args.cot,  # chain of thought
        prompting_strategy=script_args.prompting_strategy,
        continue_infer=continue_results,
    )
