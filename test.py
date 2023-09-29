import os

import pandas as pd
from dataset import DatasetWrapper

from model import get_model
from pipelines import EvalPipeline
from script_arguments import ScriptArguments
from torch.utils.data import DataLoader

from transformers import HfArgumentParser
from utils import save_to_json, set_seed

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)

    # Load dataset (you can process it here)
    dataset_wrapper = DatasetWrapper(
        dataset_name=script_args.dataset_name,
        prompting_strategy=script_args.prompting_strategy,
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

    # Load model
    model, tokenizer = get_model(config=script_args)
    model.eval()

    eval_pipeline = EvalPipeline(
        task=dataset_wrapper.task, model=model, tokenizer=tokenizer
    )

    # Save results
    if not os.path.exists(script_args.output_dir):
        os.makedirs(script_args.output_dir)

    ds_exact_name = (
        script_args.dataset_name.split("/")[-1]
        + "_"
        + script_args.model_name.split("/")[-1]
        + f"_pt{script_args.prompting_strategy}"
        + ("_fewshot" if script_args.fewshot_prompting else "")
        + ("_randchoice" if script_args.random_mtpc else "")
        + f"_seed{script_args.seed}"
    )

    json_file = os.path.join(script_args.output_dir,
                             f"generations_{ds_exact_name}.json")
    metric_file = os.path.join(script_args.output_dir,
                               f"metrics_{ds_exact_name}.json")

    if script_args.continue_infer:
        if os.path.exists(json_file):
            df1 = pd.read_json(json_file)
            start_idx = len(df1)
        else:
            raise FileNotFoundError(
                f"File {json_file} does not exist! Terminating...")
    else:
        start_idx = 0

    # Evaluate
    def save_results(generations, metrics=None):
        if script_args.continue_infer and os.path.exists(json_file):
            df2 = pd.DataFrame(generations)
            df3 = df1.append(df2, ignore_index=True)
            generations = df3.to_dict(orient="list")

        save_to_json(
            generations,
            json_file
        )
        if metrics is not None:
            save_to_json(
                metrics,
                metric_file
            )

    eval_pipeline.run(
        ds_wrapper=dataset_wrapper,
        ds_loader=dataset_loader,
        saving_fn=save_results,
        start_idx=start_idx,
        few_shot=script_args.fewshot_prompting,  # few-shot prompting
        random_mtpc=script_args.random_mtpc,     # random multiple choice
        cot=script_args.cot,                     # chain of thought
        prompting_strategy=script_args.prompting_strategy,
    )
