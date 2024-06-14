import json
from typing import Dict
import os
import pandas as pd
from nltk.metrics.scores import f_measure


def normalize_text(text: str, keep_punc=False) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    if keep_punc:
        text = white_space_fix(lower(text))
    else:
        text = white_space_fix(remove_punc(lower(text)))

    if len(text) == 0:
        text = "."

    return text


def read_json_file(filepath: str) -> Dict:
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf8") as f:
            data = json.load(f)
    else:
        data = {}

    return data


def save_to_json(data: Dict, filename: str, outdir: str = "./"):
    filepath = os.path.join(outdir, filename)
    old_data = read_json_file(filepath)
    old_data.update(data)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(old_data, f, ensure_ascii=False, indent=4)


def save_to_csv(data: Dict, filename: str, outdir: str = "./"):
    filepath = os.path.join(outdir, filename)
    data.to_csv(filepath, mode="a")


def save_to_xlsx(data, filename: str):
    writer = pd.ExcelWriter(filename)
    _ = [
        sheet_data.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
        for sheet_name, sheet_data in data.items()
    ]
    writer.close()


def info_from_filename(filename):
    info_arr = filename[:-5].split("_")

    tasks = {
        "question-answering": {
            "xquad_xtreme": "xQUAD EXTREME",
            "mlqa": "MLQA",
        },
        "summarization": {
            "vietnews": "VietNews",
            "wikilingua": "WikiLingua",
        },
        "text-classification": {
            "vsmec": "VSMEC",
            "phoatis": "PhoATIS",
        },
        "toxicity-detection": {
            "victsd": "UIT-ViCTSD",
            "vihsd": "UIT-ViHSD",
        },
        "translation": {
            "phomt-envi": "PhoMT English-Vietnamese",
            "phomt-vien": "PhoMT Vietnamese-English",
            "opus100-envi": "OPUS-100 English-Vietnamese",
            "opus100-vien": "OPUS-100 Vietnamese-English",
        },
        "sentiment-analysis": {
            "vlsp": "VLSP 2016",
            "vsfc": "UIT-VSFC",
        },
        "informationretrieval": {
            "mmarco": "mmarco",
            "mrobust": "mrobust",
        },
        "knowledge": {
            "zaloe2e": "ZaloE2E",
            "vimmrc": "ViMMRC",
        },
        "language-modelling": {
            "mlqa-mlm": "MLQA",
            "vsec": "VSEC",
        },
        "reasoning": {
            "math-azr": "MATH Level 1 - Azure",
            "math-gcp": "MATH Level 1 - Google Cloud",
            "srnatural-azr": "Synthetic Reasoning (Natural) - Azure",
            "srnatural-gcp": "Synthetic Reasoning (Natural) - Google Cloud",
            "srabstract-azr": "Synthetic Reasoning (Abstract Symbol)- Azure",
            "srabstract-gcp": "Synthetic Reasoning (Abstract Symbol)- Google Cloud",
        },
    }

    dataset_ids = {
        "math_level1_azr": "math-azr",
        "math_level1_gcp": "math-gcp",
        "xquad_xtreme": "xquad_xtreme",
        "mlqa_MLM": "mlqa-mlm",
        "VSEC": "vsec",
        "mlqa": "mlqa",
        "vietnews": "vietnews",
        "wiki_lingua": "wikilingua",
        "VSMEC": "vsmec",
        "PhoATIS": "phoatis",
        "ViCTSD": "victsd",
        "ViHSD": "vihsd",
        "PhoMT_envi": "phomt-envi",
        "PhoMT_vien": "phomt-vien",
        "opus100_envi": "opus100-envi",
        "opus100_vien": "opus100-vien",
        "vlsp2016": "vlsp",
        "UIT-VSFC": "vsfc",
        "mmarco": "mmarco",
        "mrobust": "mrobust",
        "zalo_e2eqa": "zaloe2e",
        "ViMMRC": "vimmrc",
        "synthetic_natural_azr": "srnatural-azr",
        "synthetic_natural_gcp": "srnatural-gcp",
        "synthetic_induction_azr": "srabstract-azr",
        "synthetic_induction_gcp": "srabstract-gcp",
        "synthetic_pattern_match_azr": "srabstract-azr",
        "synthetic_pattern_match_gcp": "srabstract-gcp",
        "synthetic_variable_substitution_azr": "srabstract-azr",
        "synthetic_variable_substitution_gcp": "srabstract-gcp",
        # "srabstract-gcp": "srabstract-gcp",
    }

    dataset_id = None
    for dataset_filename in dataset_ids:
        if dataset_filename in filename:
            dataset_id = dataset_ids[dataset_filename]
            break

    setting_id = None
    for setting_id_i in [
        "fairness",
        "robustness",
        "randchoice",
        "fewshot_cot",
        "fewshot",
        "pt0",
        "pt1",
        "pt2",
        "pt3",
    ]:
        if setting_id_i in filename:
            setting_id = setting_id_i
            if setting_id == "fewshot":
                setting_id = "fs"

            if setting_id == "fewshot_cot":
                setting_id = "cot"
            break

    model_id = None
    for info in info_arr:
        if info.startswith(
            (
                "ura",
                "Llama-2",
                "PhoGPT",
                "gpt-3.5-turbo",
                "gpt-4",
                "vietcuna",
                "MixSUra",
                "gemini",
                "Vistral",
                "GemSUra",
            )
        ):
            model_id = info
            break

    task_id = None
    for task_name in tasks:
        if dataset_id in tasks[task_name]:
            task_id = task_name
            break

    seed = info_arr[-1]

    return task_id, dataset_id, model_id, setting_id, seed
