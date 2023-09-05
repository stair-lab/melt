import json
import os

import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

from model import get_model
from pipelines import EvalPipeline
from dataset import DatasetWrapper
from script_arguments import ScriptArguments


def save_to_json(data, name):
    jsonString = json.dumps(data, indent=4)
    jsonFile = open(name, "w")
    jsonFile.write(jsonString)
    jsonFile.close()


def save_to_csv(data, name):
    df = pd.DataFrame(data)
    df.to_csv(name, index=False)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Load dataset (you can process it here)
    dataset_wrapper = DatasetWrapper(
        dataset_name=script_args.dataset_name,
        prompting_strategy=script_args.prompting_strategy,
    )
    dataset_loader = DataLoader(
        dataset_wrapper.get_dataset(),
        batch_size=script_args.per_device_eval_batch_size,
        shuffle=False,
    )

    # Load model
    model, tokenizer = get_model(config=script_args)
    model.eval()

    eval_pipeline = Pipeline(
        task=dataset_wrapper.task, model=model, tokenizer=tokenizer
    )

    # Save results
    if not os.path.exists(script_args.output_dir):
        os.makedirs(script_args.output_dir)

    # Evaluate
    def save_results(generations, results=None):
        ds_exact_name = (
            script_args.dataset_name.split("/")[-1]
            + "_"
            + script_args.model_name.split("/")[-1]
            + f"_pt{script_args.prompting_strategy}"
        )
        save_to_csv(
            generations,
            os.path.join(script_args.output_dir,
                         f"results_{ds_exact_name}.csv"),
        )
        if results is not None:
            save_to_json(
                results,
                os.path.join(script_args.output_dir,
                             f"results_{ds_exact_name}.json"),
            )

    eval_pipeline.run(
        ds_wrapper=dataset_wrapper, ds_loader=dataset_loader, saving_fn=save_results
    )
