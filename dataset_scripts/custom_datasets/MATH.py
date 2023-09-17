import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import sys


service = sys.argv[1]
if service == "gcloud":
    from gcloud_utils import translate_text
    prefix = "GCP"
elif service == "azure":
    from azure_utils import translate_text
    prefix = "AZR"
else:
    raise ValueError("Service must be 'gcloud' or 'azure'")


def create_dataset(training=False):
    split = "train" if training else "test"
    m1 = load_dataset("lighteval/MATH", "algebra", split=split)
    m2 = load_dataset("lighteval/MATH",
                      "counting_and_probability", split=split)
    m3 = load_dataset("lighteval/MATH", "geometry", split=split)
    m4 = load_dataset("lighteval/MATH", "intermediate_algebra", split=split)
    m5 = load_dataset("lighteval/MATH", "number_theory", split=split)
    m6 = load_dataset("lighteval/MATH", "prealgebra", split=split)
    m7 = load_dataset("lighteval/MATH", "precalculus", split=split)

    dataset = {'problem': [], 'level': [], 'type': [], 'solution': []}
    save_path = 'MATH_training' if training else 'MATH'

    for header in dataset.keys():
        for i in range(1, 8):
            dataset[header].extend(eval(f"m{i}")[header])
    df = pd.DataFrame(dataset)
    df.to_csv(f'./Original/{save_path}_original.csv', index=False)

    # Translate dataset problem and solution from English to Vietnamese
    # with batch size 10
    batch_size = 10
    new_problem = []
    new_solution = []
    for i in tqdm(range(0, len(dataset["problem"]), batch_size)):
        response_problem = translate_text(
            list_text=dataset["problem"][i:i+batch_size], project_id="ura-llama")
        new_problem.extend(response_problem)

        response_solution = translate_text(
            list_text=dataset["solution"][i:i+batch_size], project_id="ura-llama")
        new_solution.extend(response_solution)

    dataset["problem"] = new_problem
    dataset["solution"] = new_solution

    df = pd.DataFrame(dataset)
    df.to_csv(f'./{prefix}/{save_path}.csv', index=False)


if not os.path.exists(f'./{prefix}/MATH.csv'):
    print("Creating test dataset...")
    create_dataset(training=False)
if not os.path.exists(f'./{prefix}/MATH_training.csv'):
    print("Creating training dataset...")
    create_dataset(training=True)

# Filter only Level 1 problems
df = pd.read_csv(f'./{prefix}/MATH.csv')
df = df[df['level'] == "Level 1"]
df.to_csv(f'./{prefix}/math_level1.csv', index=False)

df = pd.read_csv(f'./{prefix}/MATH_training.csv')
df = df[df['level'] == "Level 1"]
df.to_csv(f'./{prefix}/math_level1_training.csv', index=False)
