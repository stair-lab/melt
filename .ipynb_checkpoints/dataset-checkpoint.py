import random
from datasets import load_dataset
from prompt_template import CALIBRATION_INSTRUCTION, PROMPT_TEMPLATE


def eval_keys(keys):
    def eval_x(x):
        if isinstance(keys, str):
            x[keys] = eval(x[keys])
        elif isinstance(keys, list):
            for key in keys:
                x[key] = eval(x[key])
        return x

    return eval_x


class DatasetWrapper:
    def __init__(self, dataset_name, prompting_strategy=0, fewshots=None) -> None:
        self.dataset_name = dataset_name
        self.prompting_strategy = prompting_strategy
        self.fewshots = fewshots
        self.dataset_training = None
        self.dataset_testing = None

        self.get_dataset_config()
        self.get_prompt()

    def get_prompt(self):
        if self.prompting_strategy not in [0, 1, 2, 3]:
            raise ValueError("Prompting strategy is not supported")
        task = self.task.split("_")[0]
        self.prompt = PROMPT_TEMPLATE[task][self.prompting_strategy]
        if task in CALIBRATION_INSTRUCTION:
            self.calibration_prompt = CALIBRATION_INSTRUCTION[task][
                self.prompting_strategy
            ]
        else:
            self.calibration_prompt = None

    def get_dataset_config(self):
        # Question Answering
        if self.dataset_name == "VIMQA":
            self.task = "question-answering"
            pass

        elif self.dataset_name == "xquad_xtreme":
            self.task = "question-answering"
            self.dataset_testing = load_dataset(
                "juletxara/xquad_xtreme", "vi", split="test"
            )
            self.context = "context"
            self.question = "question"
            self.answer = "answers"

        elif self.dataset_name == "xquad_xtreme_robustness":
            self.task = "question-answering"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Robustness/xquad_xtreme_for_robustness.csv",
                split="train",
            )
            self.dataset_testing = self.dataset_testing.map(
                eval_keys("answers"))
            self.context = "context"
            self.question = "question"
            self.answer = "answers"

        elif self.dataset_name == "xquad_xtreme_fairness":
            self.task = "question-answering"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Fairness/xquad_xtreme_for_fairness.csv",
                split="train",
            )
            self.dataset_testing = self.dataset_testing.map(
                eval_keys("answers"))
            self.context = "context"
            self.question = "question"
            self.answer = "answers"

        elif self.dataset_name == "mlqa":
            self.task = "question-answering"
            self.dataset_testing = load_dataset(
                self.dataset_name, "mlqa.vi.vi", split="test"
            )
            self.context = "context"
            self.question = "question"
            self.answer = "answers"

        elif self.dataset_name == "mlqa_robustness":
            self.task = "question-answering"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Robustness/mlqa_for_robustness.csv",
                split="train",
            )
            self.dataset_testing = self.dataset_testing.map(
                eval_keys("answers"))
            self.context = "context"
            self.question = "question"
            self.answer = "answers"

        elif self.dataset_name == "mlqa_fairness":
            self.task = "question-answering"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Fairness/mlqa_for_fairness.csv",
                split="train",
            )
            self.dataset_testing = self.dataset_testing.map(
                eval_keys("answers"))
            self.context = "context"
            self.question = "question"
            self.answer = "answers"

        # Summarization
        elif self.dataset_name == "vietnews":
            self.task = "summarization"
            self.dataset_testing = load_dataset(
                "Yuhthe/vietnews", split="test")
            self.dataset_testing.set_format(columns=["article", "abstract"])
            self.original_text = "article"
            self.summarized_text = "abstract"

        elif self.dataset_name == "vietnews_robustness":
            self.task = "summarization"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Robustness/vietnews_for_robustness.csv",
                split="train",
            )
            self.dataset_testing.set_format(columns=["article", "abstract"])
            self.original_text = "article"
            self.summarized_text = "abstract"

        elif self.dataset_name == "wiki_lingua":
            self.task = "summarization"
            self.dataset_testing = load_dataset(
                "GEM/wiki_lingua", "vi", split="test")
            self.original_text = "source"
            self.summarized_text = "target"

        elif self.dataset_name == "wiki_lingua_robustness":
            self.task = "summarization"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Robustness/wiki_lingua_for_robustness.csv",
                split="train",
            )
            self.original_text = "source"
            self.summarized_text = "target"

        # Sentiment Analysis
        elif self.dataset_name == "UIT-VSFC":
            self.task = "sentiment-analysis"
            self.dataset_testing = load_dataset(
                "csv", data_files="datasets/Original/UIT-VSFC.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv",
                data_files="datasets/Training/UIT-VSFC_training.csv",
                split="train",
            )
            self.text = "text"
            self.label = "label"

        elif self.dataset_name == "UIT-VSFC_robustness":
            self.task = "sentiment-analysis"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Robustness/UIT-VSFC_for_robustness.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "csv",
                data_files="datasets/Training/UIT-VSFC_training.csv",
                split="train",
            )
            self.text = "text"
            self.label = "label"

        elif self.dataset_name == "UIT-VSFC_fairness":
            self.task = "sentiment-analysis"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Fairness/UIT-VSFC_for_fairness.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "csv",
                data_files="datasets/Training/UIT-VSFC_training.csv",
                split="train",
            )
            self.text = "text"
            self.label = "label"

        elif self.dataset_name == "vlsp2016":
            self.task = "sentiment-analysis"
            self.dataset_testing = load_dataset(
                "csv", data_files="datasets/Original/vlsp2016.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv",
                data_files="datasets/Training/vlsp2016_training.csv",
                split="train",
            )
            self.text = "Data"
            self.label = "Class"

        elif self.dataset_name == "vlsp2016_robustness":
            self.task = "sentiment-analysis"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Robustness/vlsp2016_for_robustness.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "csv",
                data_files="datasets/Training/vlsp2016_training.csv",
                split="train",
            )
            self.text = "Data"
            self.label = "Class"

        elif self.dataset_name == "vlsp2016_fairness":
            self.task = "sentiment-analysis"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Fairness/vlsp2016_for_fairness.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "csv",
                data_files="datasets/Training/vlsp2016_training.csv",
                split="train",
            )
            self.text = "Data"
            self.label = "Class"

        # Text Classification
        elif self.dataset_name == "PhoATIS":
            self.task = "text-classification-atis"
            self.dataset_testing = load_dataset(
                "csv", data_files="datasets/Original/PhoATIS.csv", split="train"
            )
            # self.dataset_testing = self.dataset_testing.map(eval_keys("label"))
            self.dataset_training = load_dataset(
                "csv",
                data_files="datasets/Training/PhoATIS_training.csv",
                split="train",
            )
            self.dataset_training = self.dataset_training.map(
                eval_keys("label"))
            self.text = "sentence"
            self.label = "label"

        elif self.dataset_name == "PhoATIS_robustness":
            self.task = "text-classification-atis"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Robustness/PhoATIS_for_robustness.csv",
                split="train",
            )
            # self.dataset_testing = self.dataset_testing.map(eval_keys("label"))
            self.dataset_training = load_dataset(
                "csv",
                data_files="datasets/Training/PhoATIS_training.csv",
                split="train",
            )
            self.dataset_training = self.dataset_training.map(
                eval_keys("label"))
            self.text = "sentence"
            self.label = "label"

        elif self.dataset_name == "PhoATIS_fairness":
            self.task = "text-classification-atis"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Fairness/PhoATIS_for_fairness.csv",
                split="train",
            )
            # self.dataset_testing = self.dataset_testing.map(eval_keys("label"))
            self.dataset_training = load_dataset(
                "csv",
                data_files="datasets/Training/PhoATIS_training.csv",
                split="train",
            )
            self.dataset_training = self.dataset_training.map(
                eval_keys("label"))
            self.text = "sentence"
            self.label = "label"

        elif self.dataset_name == "UIT-VSMEC":
            self.task = "text-classification-vsmec"
            self.dataset_testing = load_dataset(
                "csv", data_files="datasets/Original/UIT-VSMEC.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv",
                data_files="datasets/Training/UIT-VSMEC_training.csv",
                split="train",
            )
            self.text = "Sentence"
            self.label = "Label"

        elif self.dataset_name == "UIT-VSMEC_robustness":
            self.task = "text-classification-vsmec"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Robustness/UIT-VSMEC_for_robustness.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "csv",
                data_files="datasets/Training/UIT-VSMEC_training.csv",
                split="train",
            )
            self.text = "Sentence"
            self.label = "Label"

        elif self.dataset_name == "UIT-VSMEC_fairness":
            self.task = "text-classification-vsmec"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Fairness/UIT-VSMEC_for_fairness.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "csv",
                data_files="datasets/Training/UIT-VSMEC_training.csv",
                split="train",
            )
            self.text = "Sentence"
            self.label = "Label"

        # Knowledge
        elif self.dataset_name == "zalo_e2eqa":
            self.task = "knowledge-openended"
            self.dataset_testing = load_dataset(
                "csv", data_files="datasets/Original/zalo_e2eqa.csv", split="train"
            )
            if self.fewshots is None:
                selected_sample_idx = list(
                    random.sample(range(len(self.dataset_testing)), 5)
                )
                self.dataset_training = [
                    self.dataset_testing[s] for s in selected_sample_idx
                ]
            else:
                self.dataset_training = self.fewshots
                # Get index of selected samples
                selected_sample_idx = [
                    self.dataset_testing[(self.dataset_testing['id'] == s['id'])].index.item() for s in self.dataset_training
                ]

            self.dataset_testing = self.dataset_testing.select(
                [
                    i
                    for i in range(len(self.dataset_testing))
                    if i not in selected_sample_idx
                ]
            )
            self.question = "question"
            self.answer = "answers"

        elif self.dataset_name == "zalo_e2eqa_robustness":
            self.task = "knowledge-openended"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Robustness/zalo_e2eqa_for_robustness.csv",
                split="train",
            )
            if self.fewshots is None:
                selected_sample_idx = list(
                    random.sample(range(len(self.dataset_testing)), 5)
                )
                self.dataset_training = [
                    self.dataset_testing[s] for s in selected_sample_idx
                ]
            else:
                self.dataset_training = self.fewshots
                # Get index of selected samples
                selected_sample_idx = [
                    self.dataset_testing[(self.dataset_testing['id'] == s['id'])].index.item() for s in self.dataset_training
                ]
            self.dataset_testing = self.dataset_testing.select(
                [
                    i
                    for i in range(len(self.dataset_testing))
                    if i not in selected_sample_idx
                ]
            )
            self.question = "question"
            self.answer = "answers"

        elif self.dataset_name == "ViMMRC":
            self.task = "knowledge-mtpchoice"
            self.dataset_testing = load_dataset(
                "csv", data_files="datasets/Original/ViMMRC.csv", split="train"
            )
            self.dataset_testing = self.dataset_testing.map(
                eval_keys("options"))
            self.dataset_training = load_dataset(
                "csv", data_files="datasets/Training/ViMMRC_training.csv", split="train"
            )
            self.dataset_training = self.dataset_training.map(
                eval_keys("options"))
            self.context = "article"
            self.question = "question"
            self.options = "options"
            self.answer = "answer"

        elif self.dataset_name == "ViMMRC_robustness":
            self.task = "knowledge-mtpchoice"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Robustness/ViMMRC_for_robustness.csv",
                split="train",
            )
            self.dataset_testing = self.dataset_testing.map(
                eval_keys("options"))
            self.dataset_training = load_dataset(
                "csv", data_files="datasets/Training/ViMMRC_training.csv", split="train"
            )
            self.dataset_training = self.dataset_training.map(
                eval_keys("options"))
            self.context = "article"
            self.question = "question"
            self.options = "options"
            self.answer = "answer"

        # Toxicity Detection
        elif self.dataset_name == "ViCTSD":
            self.task = "toxicity-detection-ViCTSD"
            self.dataset_testing = load_dataset(
                "tarudesu/ViCTSD", split="test")
            self.dataset_training = load_dataset(
                "tarudesu/ViCTSD", split="train")
            self.text = "Comment"
            self.label = "Toxicity"

        elif self.dataset_name == "ViCTSD_robustness":
            self.task = "toxicity-detection-ViCTSD"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Robustness/ViCTSD_for_robustness.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "tarudesu/ViCTSD", split="train")
            self.text = "Comment"
            self.label = "Toxicity"

        elif self.dataset_name == "ViCTSD_fairness":
            self.task = "toxicity-detection-ViCTSD"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Fairness/ViCTSD_for_fairness.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "tarudesu/ViCTSD", split="train")
            self.text = "Comment"
            self.label = "Toxicity"

        elif self.dataset_name == "ViHSD":
            self.task = "toxicity-detection-ViHSD"
            self.dataset_testing = load_dataset(
                "csv", data_files="datasets/Original/ViHSD.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="datasets/Training/ViHSD_training.csv", split="train"
            )
            self.text = "free_text"
            self.label = "label_id"

        elif self.dataset_name == "ViHSD_robustness":
            self.task = "toxicity-detection-ViHSD"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Robustness/ViHSD_for_robustness.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "csv", data_files="datasets/Training/ViHSD_training.csv", split="train"
            )
            self.text = "free_text"
            self.label = "label_id"

        elif self.dataset_name == "ViHSD_fairness":
            self.task = "toxicity-detection-ViHSD"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Fairness/ViHSD_for_fairness.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "csv", data_files="datasets/Training/ViHSD_training.csv", split="train"
            )
            self.text = "free_text"
            self.label = "label_id"

        # Information Retrieval
        elif self.dataset_name == "mmarco":
            self.task = "information-retrieval_mmarco"
            self.dataset_testing = load_dataset(
                "json", data_files="datasets/Original/mmarco.json", split="train"
            )
            self.dataset_training = load_dataset(
                "json",
                data_files="datasets/Training/mmarco_training.json",
                split="train",
            )
            self.id = "id"
            self.query = "query"
            self.passage = "passages"
            self.answer = "references"

        elif self.dataset_name == "mmarco_robustness":
            self.task = "information-retrieval_mmarco"
            self.dataset_testing = load_dataset(
                "json",
                data_files="datasets/Robustness/mmarco_for_robustness.json",
                split="train",
            )
            self.dataset_training = load_dataset(
                "json",
                data_files="datasets/Training/mmarco_training.json",
                split="train",
            )
            self.id = "id"
            self.query = "query"
            self.passage = "passages"
            self.answer = "references"

        elif self.dataset_name == "mmarco_fairness":
            self.task = "information-retrieval_mmarco"
            self.dataset_testing = load_dataset(
                "json",
                data_files="datasets/Fairness/mmarco_for_fairness.json",
                split="train",
            )
            self.dataset_training = load_dataset(
                "json",
                data_files="datasets/Training/mmarco_training.json",
                split="train",
            )
            self.id = "id"
            self.query = "query"
            self.passage = "passages"
            self.answer = "references"

        elif self.dataset_name == "mrobust":
            self.task = "information-retrieval"
            self.dataset_testing = load_dataset(
                "json", data_files="datasets/Original/mrobust.json", split="train"
            )
            self.dataset_training = load_dataset(
                "json",
                data_files="datasets/Training/mmarco_training.json",
                split="train",
            )
            self.id = "id"
            self.query = "query"
            self.passage = "passages"
            self.answer = "references"

        elif self.dataset_name == "mrobust_robustness":
            self.task = "information-retrieval"
            self.dataset_testing = load_dataset(
                "json",
                data_files="datasets/Robustness/mrobust_for_robustness.json",
                split="train",
            )
            self.dataset_training = load_dataset(
                "json",
                data_files="datasets/Training/mmarco_training.json",
                split="train",
            )
            self.id = "id"
            self.query = "query"
            self.passage = "passages"
            self.answer = "references"

        elif self.dataset_name == "mrobust_fairness":
            self.task = "information-retrieval"
            self.dataset_testing = load_dataset(
                "json",
                data_files="datasets/Fairness/mrobust_for_fairness.json",
                split="train",
            )
            self.dataset_training = load_dataset(
                "json",
                data_files="datasets/Training/mmarco_training.json",
                split="train",
            )
            self.id = "id"
            self.query = "query"
            self.passage = "passages"
            self.answer = "references"

        # Language
        elif self.dataset_name == "mlqa_MLM":
            self.task = "language-modelling-filling"
            self.dataset_testing = load_dataset(
                "csv", data_files="datasets/Original/mlqa_MLM.csv", split="train"
            )
            if self.fewshots is None:
                selected_sample_idx = list(
                    random.sample(range(len(self.dataset_testing)), 3)
                )
                self.dataset_training = [
                    self.dataset_testing[s] for s in selected_sample_idx
                ]
            else:
                self.dataset_training = self.fewshots
                # Get index of selected samples
                selected_sample_idx = [
                    self.dataset_testing[
                        (self.dataset_testing['masked'] == s['masked']) &
                        (self.dataset_testing['context'] == s['context'])
                    ].index.item() for s in self.dataset_training
                ]
            self.dataset_testing = self.dataset_testing.select(
                [
                    i
                    for i in range(len(self.dataset_testing))
                    if i not in selected_sample_idx
                ]
            )
            self.source = "masked"
            self.target = "context"

        elif self.dataset_name == "mlqa_MLM_fairness":
            self.task = "language-modelling-filling"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Fairness/mlqa_MLM_for_fairness.csv",
                split="train",
            )
            if self.fewshots is None:
                selected_sample_idx = list(
                    random.sample(range(len(self.dataset_testing)), 3)
                )
                self.dataset_training = [
                    self.dataset_testing[s] for s in selected_sample_idx
                ]
            else:
                self.dataset_training = self.fewshots
                # Get index of selected samples
                selected_sample_idx = [
                    self.dataset_testing[
                        (self.dataset_testing['masked'] == s['masked']) &
                        (self.dataset_testing['context'] == s['context'])
                    ].index.item() for s in self.dataset_training
                ]
            self.dataset_testing = self.dataset_testing.select(
                [
                    i
                    for i in range(len(self.dataset_testing))
                    if i not in selected_sample_idx
                ]
            )
            self.source = "masked"
            self.target = "context"

        elif self.dataset_name == "VSEC":
            self.task = "language-modelling-correction"
            self.dataset_testing = load_dataset(
                "csv", data_files="datasets/Original/VSEC.csv", split="train"
            )
            if self.fewshots is None:
                selected_sample_idx = list(
                    random.sample(range(len(self.dataset_testing)), 3)
                )
                self.dataset_training = [
                    self.dataset_testing[s] for s in selected_sample_idx
                ]
            else:
                self.dataset_training = self.fewshots
                # Get index of selected samples
                selected_sample_idx = [
                    self.dataset_testing[
                        (self.dataset_testing['text'] == s['text']) &
                        (self.dataset_testing['correct'] == s['correct'])
                    ].index.item() for s in self.dataset_training
                ]
            self.dataset_testing = self.dataset_testing.select(
                [
                    i
                    for i in range(len(self.dataset_testing))
                    if i not in selected_sample_idx
                ]
            )
            self.source = "text"
            self.target = "correct"

        elif self.dataset_name == "VSEC_fairness":
            self.task = "language-modelling-correction"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Fairness/VSEC_for_fairness.csv",
                split="train",
            )
            if self.fewshots is None:
                selected_sample_idx = list(
                    random.sample(range(len(self.dataset_testing)), 3)
                )
                self.dataset_training = [
                    self.dataset_testing[s] for s in selected_sample_idx
                ]
            else:
                self.dataset_training = self.fewshots
                # Get index of selected samples
                selected_sample_idx = [
                    self.dataset_testing[
                        (self.dataset_testing['text'] == s['text']) &
                        (self.dataset_testing['correct'] == s['correct'])
                    ].index.item() for s in self.dataset_training
                ]
            self.dataset_testing = self.dataset_testing.select(
                [
                    i
                    for i in range(len(self.dataset_testing))
                    if i not in selected_sample_idx
                ]
            )
            self.source = "text"
            self.target = "correct"

        # Reasoning
        elif self.dataset_name.startswith("synthetic_natural"):
            self.task = "reasoning-synthetic"
            subset = "easy_" + self.dataset_name.split("_")[-1]
            self.dataset_testing = load_dataset(
                "ura-hcmut/synthetic_reasoning_natural",
                subset,
                split="test",
            )
            self.dataset_training = load_dataset(
                "ura-hcmut/synthetic_reasoning_natural",
                subset,
                split="train",
            )
            self.source = "question"
            self.target = "target"

        elif self.dataset_name.startswith("synthetic_induction") or \
                self.dataset_name.startswith("synthetic_pattern_match") or \
                self.dataset_name.startswith("synthetic_variable_substitution"):
            self.task = "reasoning-synthetic"
            subset = '_'.join(self.dataset_name.split("_")[1:])
            self.dataset_testing = load_dataset(
                "ura-hcmut/synthetic_reasoning",
                subset,
                split="test",
            )
            self.dataset_training = load_dataset(
                "ura-hcmut/synthetic_reasoning",
                subset,
                split="train",
            )
            self.source = "source"
            self.target = "target"

        elif self.dataset_name.startswith("math_level1"):
            self.task = "reasoning-math"
            subset = self.dataset_name.split("_")[2]
            problem_type = self.dataset_name.split("_")[-1]
            self.dataset_testing = load_dataset(
                "ura-hcmut/MATH",
                subset,
                split="test",
            )
            self.dataset_testing = self.dataset_testing.filter(
                lambda x: x["type"] == problem_type and x["level"] == "Level 1"
            )
            self.dataset_training = load_dataset(
                "ura-hcmut/MATH",
                subset,
                split="train",
            )
            self.dataset_training = self.dataset_training.filter(
                lambda x: x["type"] == problem_type and x["level"] == "Level 1"
            )
            self.source = "problem"
            self.type = "type"
            self.target = "solution"
            self.short_target = "short_solution"

        # Translation
        elif self.dataset_name == "PhoMT_envi":
            self.task = "translation-envi"
            self.dataset_testing = load_dataset(
                "csv", data_files="datasets/Original/PhoMT.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="datasets/Training/PhoMT_training.csv", split="train"
            )
            self.source_language = "en"
            self.target_language = "vi"

        elif self.dataset_name == "PhoMT_vien":
            self.task = "translation-vien"
            self.dataset_testing = load_dataset(
                "csv", data_files="datasets/Original/PhoMT.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="datasets/Training/PhoMT_training.csv", split="train"
            )
            self.source_language = "vi"
            self.target_language = "en"

        elif self.dataset_name == "PhoMT_envi_robustness":
            self.task = "translation-envi"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Robustness/PhoMT_envi_for_robustness.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "csv", data_files="datasets/Training/PhoMT_training.csv", split="train"
            )
            self.source_language = "en"
            self.target_language = "vi"

        elif self.dataset_name == "PhoMT_vien_robustness":
            self.task = "translation-vien"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Robustness/PhoMT_vien_for_robustness.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "csv", data_files="datasets/Training/PhoMT_training.csv", split="train"
            )
            self.source_language = "vi"
            self.target_language = "en"

        elif self.dataset_name == "opus100_envi":
            self.task = "translation-envi"
            self.dataset_testing = load_dataset(
                "vietgpt/opus100_envi", split="test")
            self.dataset_training = load_dataset(
                "vietgpt/opus100_envi", split="train")
            self.source_language = "en"
            self.target_language = "vi"

        elif self.dataset_name == "opus100_vien":
            self.task = "translation-vien"
            self.dataset_testing = load_dataset(
                "vietgpt/opus100_envi", split="test")
            self.dataset_training = load_dataset(
                "vietgpt/opus100_envi", split="train")
            self.source_language = "vi"
            self.target_language = "en"

        elif self.dataset_name == "opus100_envi_robustness":
            self.task = "translation-envi"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Robustness/opus100_envi_for_robustness.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "vietgpt/opus100_envi", split="train")
            self.source_language = "en"
            self.target_language = "vi"

        elif self.dataset_name == "opus100_vien_robustness":
            self.task = "translation-vien"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="datasets/Robustness/opus100_vien_for_robustness.csv",
                split="train",
            )
            self.dataset_training = load_dataset(
                "vietgpt/opus100_envi", split="train")
            self.source_language = "vi"
            self.target_language = "en"

        else:
            raise ValueError("Dataset is not supported")

    def get_dataset_testing(self):
        if self.dataset_testing is None:
            raise ValueError("Dataset testing is not available")
        return self.dataset_testing

    def get_dataset_training(self):
        if self.dataset_training is None:
            raise ValueError("Dataset training is not available")
        return self.dataset_training
