import os
from .metrics.utils import read_json_file, save_to_xlsx
import argparse
import pandas as pd


metrics = {
    "accuracy_std": "Accuracy",
    "exact_match_std": "Exact Match",
    "average_exact_match_std": "Average Exact Match",
    "f1_score_std": "F1 Score",
    "roc_auc_std": "AUC ROC",
    "equality_std": "Equivalent",
    "race_profession_stereotypical_std": "Stereotypical associations (race, profession)",
    "race_profession_demographic_std": "Demographic representation (race)",
    "gender_profession_stereotypical_std": "Stereotypical associations (gender, profession)",
    "gender_profession_demographic_std": "Demographic representation (gender)",
    "toxicity_std": "Toxicity",
    "rouge1_std": "ROUGE-1",
    "rouge2_std": "ROUGE-2",
    "rougeL_std": "ROUGE-L",
    "bleu_std": "BLEU",
    "summac_std": "SummaC",
    "BERTScore-F1_std": "BERTScore-F1",
    "coverage_std": "Coverage",
    "density_std": "Density",
    "compression_std": "Compression",
    "hLepor_std": "hLEPOR",
    "cer_std": "Character Error Rate",
    "wer_std": "Word Error Rate",
    "ced_std": "Character Edit Distance",
    "wed_std": "Word Edit Distance",
    "perplexity_std": "Perplexity",
    "ece_10_bin_std": "Expected Calibration Error",
    "acc_top_10_percentile_std": "acc@10",
    "regular_mrr@10_std": "MRR@10 (Top 30)",
    "regular_ndcg@10_std": "NDCG@10 (Top 30)",
    "boosted_mrr@10_std": "MRR@10",
    "boosted_ndcg@10_std": "NDCG@10",
}

metric_ud = {
    "Accuracy": 1,
    "Average Exact Match": 1,
    "Exact Match": 1,
    "F1 Score": 1,
    "AUC ROC": 1,
    "AUC PR": 1,
    "Precision": 1,
    "Recall": 1,
    "Equivalent": 1,
    "Bias": -1,
    "Toxicity": -1,
    "ROUGE-1": 1,
    "ROUGE-2": 1,
    "ROUGE-L": 1,
    "BLEU": 1,
    "SummaC": 1,
    "BERTScore": 1,
    "Coverage": 1,
    "Density": 1,
    "Compression": 1,
    "hLEPOR": 1,
    "Character Error Rate": -1,
    "Word Error Rate": -1,
    "Character Edit Distance": -1,
    "Word Edit Distance": -1,
    "Perplexity": -1,
    "Expected Calibration Error": -1,
    "acc@10": 1,
    "MRR@10 (Top 30)": 1,
    "NDCG@10 (Top 30)": 1,
    "MRR@10": 1,
    "NDCG@10": 1,
}

tasks = {
    "Information Retrieval": "informationretrieval",
    "Knowledge": "knowledge",
    "Language Modelling": "language-modelling",
    "Question Answering": "question-answering",
    "Reasoning": "reasoning",
    "Summarization": "summarization",
    "Text Classification": "text-classification",
    "Toxicity Detection": "toxicity-detection",
    "Translation": "translation",
    "Sentiment Analysis": "sentiment-analysis",
}

settings = {
    "Normal": "",
    "Few-shot Leanring": "fs",
    "Prompt Strategy 0": "pt0",
    "Prompt Strategy 1": "pt1",
    "Prompt Strategy 2": "pt2",
    "Prompt Strategy 3": "pt3",
    "Chain-of-Thought": "cot",
    "Fairness": "fairness",
    "Robustness": "robustness",
    "Random Order Choices": "randchoice",
}

task_w_settings = {
    "Information Retrieval": ["Normal", "Few-shot Leanring", "Robustness", "Fairness"],
    "Knowledge": ["Normal", "Few-shot Leanring", "Robustness", "Random Order Choices"],
    "Language Modelling": ["Normal", "Few-shot Leanring", "Fairness"],
    "Question Answering": [
        "Prompt Strategy 0",
        "Prompt Strategy 1",
        "Prompt Strategy 2",
        "Robustness",
        "Fairness",
    ],
    "Reasoning": ["Normal", "Few-shot Leanring", "Chain-of-Thought"],
    "Summarization": [
        "Prompt Strategy 0",
        "Prompt Strategy 1",
        "Prompt Strategy 2",
        "Robustness",
    ],
    "Text Classification": ["Normal", "Few-shot Leanring", "Robustness", "Fairness"],
    "Toxicity Detection": ["Normal", "Few-shot Leanring", "Robustness", "Fairness"],
    "Translation": ["Few-shot Leanring", "Robustness"],
    "Sentiment Analysis": ["Normal", "Few-shot Leanring", "Robustness", "Fairness"],
}

datasets = {
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

models = {
    "ura-llama-7b": "URA-LLaMa 7B",
    "ura-llama-13b": "URA-LLaMa 13B",
    "ura-llama-70b": "URA-LLaMa 70B",
    "Llama-2-7b-chat-hf": "LLaMa-2 7B",
    "Llama-2-13b-chat-hf": "LLaMa-2 13B",
    "Llama-2-70b-chat-hf": "LLaMa-2 70B",
    "gpt-3.5-turbo": "GPT-3.5 Turbo",
    "gpt-4": "GPT-4",
    "vietcuna-7b-v3": "Vietcuna 7B",
    "PhoGPT-7B5-Instruct": "PhoGPT",
}


def info_from_filename(filename):
    info_arr = filename[:-5].split("_")

    dataset_ids = {
        "math-azr": "math-azr",
        "math-gcp": "math-gcp",
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
        "srnatural-azr": "srnatural-azr",
        "srnatural-gcp": "srnatural-gcp",
        "srabstract-azr": "srabstract-azr",
        "srabstract-gcp": "srabstract-gcp",
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
        "cot",
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
            break

    model_id = None
    for info in info_arr:
        if info.startswith(
            ("ura", "Llama-2", "PhoGPT", "gpt-3.5-turbo", "gpt-4", "vietcuna")
        ):
            model_id = info
            break

    seed = info_arr[-1]

    return dataset_id, model_id, setting_id, seed


def to_sheet_std(args):
    data = pd.read_excel(args.template_file, sheet_name=None, header=None)
    list_files = [
        f
        for f in os.listdir(args.out_eval_dir)
        if f.startswith("overall") and f.endswith("json")
    ]

    for filename in list_files:
        writing = False
        dataset_id, model_id, setting_id, seed = info_from_filename(filename)
        if seed != "seed42":
            continue
        if setting_id == "pt3":
            setting_id = "pt2"
        if setting_id == "pt1" and model_id == "gpt-4":
            setting_id = "pt0"
        if setting_id in ["fairness", "robustness"] and not (
            "fewshot" in filename or "pt2" in filename or "pt3" in filename
        ):
            continue

        task_id = None
        for task_id_i, dataset_ids_i in datasets.items():
            if dataset_id in dataset_ids_i:
                task_id = task_id_i

        sheet_data = None
        for task_name, task_id_i in tasks.items():
            if task_id == task_id_i:
                for setting_name_i in task_w_settings[task_name]:
                    setting_id_i = settings[setting_name_i]
                    if (
                        setting_id == setting_id_i
                        or (setting_id == "pt0" and setting_id_i == "")
                        or (
                            setting_id in ["pt1", "pt3", None]
                            and setting_id_i in ["", "pt0"]
                            and model_id in ["gpt-3.5-turbo", "vietcuna-7b-v3"]
                        )
                    ):
                        sheet_name = (
                            f"{task_id}-{setting_id_i}" if setting_id_i else task_id
                        )
                        sheet_data = data[sheet_name]
                        break

        if sheet_data is None:
            print(filename, task_id, setting_id, model_id)
            continue

        row_ids = []
        for i, row in sheet_data.iterrows():
            if "Models/" in row[0]:
                row_ids.append(i)
        row_ids.append(len(sheet_data))

        atributes = []
        sub_id = None
        dataset_data = None

        for i in range(len(row_ids) - 1):
            dataset_id_i = sheet_data.iloc[row_ids[i]][0].split("/")[-1]
            if dataset_id == dataset_id_i:
                dataset_data = sheet_data.iloc[row_ids[i] + 1 : row_ids[i + 1]]
                atributes = list(sheet_data.iloc[row_ids[i]][1:])
                sub_id = row_ids[i] + 1
                break

        model_name = models[model_id]
        for i in range(len(dataset_data)):
            model_name_i = dataset_data.iloc[i][0]
            if model_name_i == model_name:
                score_filepath = os.path.join(args.out_eval_dir, filename)
                results = read_json_file(score_filepath)
                for key, score in results.items():
                    name = metrics.get(key, None)

                    if name in atributes:
                        writing = True
                        row_score = i + sub_id
                        col_score = 1 + atributes.index(name)
                        data[sheet_name].iloc[row_score][col_score] = (
                            round(score, 4) if score else score
                        )

                break

        if not writing:
            print(sheet_name, dataset_id_i, model_name, atributes, results)

    save_to_xlsx(data, args.out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--out_eval_dir",
        default="./out_new",
        type=str,
        help="The predictions json file path",
    )
    parser.add_argument(
        "--template_file",
        default="evaluation_results_template.xlsx",
        type=str,
        help="The predictions json file path",
    )
    parser.add_argument(
        "--out_file",
        default="evaluation_results_std.xlsx",
        type=str,
        help="The output folder to save bias score",
    )
    args = parser.parse_args()
    main(args)
