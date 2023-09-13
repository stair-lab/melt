import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from gcloud_utils import translate_text


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

    for header in dataset.keys():
        for i in range(1, 8):
            dataset[header].extend(eval(f"m{i}")[header])

    # Translate dataset problem and solution from English to Vietnamese
    # with batch size 10
    batch_size = 10
    new_problem = []
    new_solution = []
    for i in tqdm(range(0, len(dataset["problem"]), batch_size)):
        response = translate_text(
            list_text=dataset["problem"][i:i+batch_size], project_id="ura-llama")
        for translation in response.translations:
            new_problem.append(translation.translated_text)

        response = translate_text(
            list_text=dataset["solution"][i:i+batch_size], project_id="ura-llama")
        for translation in response.translations:
            new_solution.append(translation.translated_text)

    dataset["problem"] = new_problem
    dataset["solution"] = new_solution

    df = pd.DataFrame(dataset)
    save_path = './MATH_training.csv' if training else './MATH.csv'
    df.to_csv(save_path, index=False)


if not os.path.exists('./MATH.csv'):
    print("Creating test dataset...")
    create_dataset(training=False)
if not os.path.exists('./MATH_training.csv'):
    print("Creating training dataset...")
    create_dataset(training=True)
