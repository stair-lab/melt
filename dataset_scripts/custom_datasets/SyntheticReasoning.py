import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from gcloud_utils import translate_text


def create_dataset(training=False):
    split = "train" if training else "test"
    m1 = load_dataset("lighteval/synthetic_reasoning_natural",
                      'easy', split=split)
    m2 = load_dataset("lighteval/synthetic_reasoning",
                      "induction", split=split)
    m3 = load_dataset("lighteval/synthetic_reasoning",
                      "pattern_match", split=split)
    m4 = load_dataset("lighteval/synthetic_reasoning",
                      "variable_substitution", split=split)

    # Translate dataset source and target from English to Vietnamese
    # with batch size 10
    batch_size = 32
    list_name = ["synthetic_reasoning_natural",
                 "synthetic_reasoning_induction",
                 "synthetic_reasoning_pattern_match",
                 "synthetic_reasoning_variable_substitution"]

    for i in range(1, 5):
        save_path = f'./{list_name[i-1]}_training.csv' if training else f'./{list_name[i-1]}.csv'
        if os.path.exists(save_path):
            continue

        source_key = "question" if i == 1 else "source"
        dataset = {source_key: eval(f'm{i}')[source_key],
                   'target': eval(f'm{i}')['target']}
        new_source = []
        new_target = []
        for i in tqdm(range(0, len(dataset[source_key]), batch_size)):
            response = translate_text(
                list_text=dataset[source_key][i:i+batch_size], project_id="ura-llama")
            for translation in response.translations:
                new_source.append(translation.translated_text)

            response = translate_text(
                list_text=dataset["target"][i:i+batch_size], project_id="ura-llama")
            for translation in response.translations:
                new_target.append(translation.translated_text)

        dataset[source_key] = new_source
        dataset["target"] = new_target

        df = pd.DataFrame(dataset)
        df.to_csv(save_path, index=False)


if not os.path.exists('./synthetic_reasoning_natural.csv') or \
        not os.path.exists('./synthetic_reasoning_induction.csv') or \
    not os.path.exists('./synthetic_reasoning_pattern_match.csv') or \
        not os.path.exists('./synthetic_reasoning_variable_substitution.csv'):
    print("Creating test dataset...")
    create_dataset(training=False)
if not os.path.exists('./synthetic_reasoning_natural_training.csv') or \
        not os.path.exists('./synthetic_reasoning_induction_training.csv') or \
    not os.path.exists('./synthetic_reasoning_pattern_match_training.csv') or \
        not os.path.exists('./synthetic_reasoning_variable_substitution_training.csv'):
    print("Creating training dataset...")
    create_dataset(training=True)
