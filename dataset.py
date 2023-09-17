from datasets import load_dataset
from prompt_template import PROMPT_TEMPLATE, CALIBRAION_INSTRUCTION


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
    def __init__(self, dataset_name, prompting_strategy=0) -> None:
        self.dataset_name = dataset_name
        self.prompting_strategy = prompting_strategy
        self.dataset_training = None
        self.dataset_testing = None

        self.get_dataset_config()
        self.get_prompt()

    def get_prompt(self):
        if self.prompting_strategy not in [0, 1, 2]:
            raise ValueError("Prompting strategy is not supported")

        self.prompt = PROMPT_TEMPLATE[self.task][self.prompting_strategy]
        if self.task in CALIBRAION_INSTRUCTION:
            self.calibration_prompt = CALIBRAION_INSTRUCTION[self.task][0]
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
                "juletxara/xquad_xtreme", "vi", split="test")
            self.context = "context"
            self.question = "question"
            self.answer = "answers"

        elif self.dataset_name == "xquad_xtreme_robustness":
            self.task = "question-answering"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="evaluation_datasets/xquad_xtreme_for_robustness.csv",
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
                data_files="evaluation_datasets/xquad_xtreme_for_fairness.csv",
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
                self.dataset_name, "mlqa.vi.vi", split="test")
            self.context = "context"
            self.question = "question"
            self.answer = "answers"

        elif self.dataset_name == "mlqa_robustness":
            self.task = "question-answering"
            self.dataset_testing = load_dataset(
                "csv",
                data_files="evaluation_datasets/mlqa_for_robustness.csv",
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
                data_files="evaluation_datasets/mlqa_for_fairness.csv",
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
                data_files="evaluation_datasets/vietnews_for_robustness.csv",
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
                data_files="evaluation_datasets/wiki_lingua_for_robustness.csv",
                split="train",
            )
            self.original_text = "source"
            self.summarized_text = "target"

        # Sentiment Analysis
        elif self.dataset_name == "UIT-VSFC":
            self.task = "sentiment-analysis"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/UIT-VSFC.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/UIT-VSFC_training.csv", split="train"
            )
            self.text = "text"
            self.label = "label"

        elif self.dataset_name == "UIT-VSFC_robustness":
            self.task = "sentiment-analysis"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/UIT-VSFC_for_robustness.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/UIT-VSFC_training.csv", split="train"
            )
            self.text = "text"
            self.label = "label"

        elif self.dataset_name == "UIT-VSFC_fairness":
            self.task = "sentiment-analysis"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/UIT-VSFC_for_fairness.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/UIT-VSFC_training.csv", split="train"
            )
            self.text = "text"
            self.label = "label"

        elif self.dataset_name == "vlsp2016":
            self.task = "sentiment-analysis"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/vlsp2016.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/vlsp2016_training.csv", split="train"
            )
            self.text = "Data"
            self.label = "Class"

        elif self.dataset_name == "vlsp2016_robustness":
            self.task = "sentiment-analysis"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/vlsp2016_for_robustness.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/vlsp2016_training.csv", split="train"
            )
            self.text = "Data"
            self.label = "Class"

        elif self.dataset_name == "vlsp2016_fairness":
            self.task = "sentiment-analysis"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/vlsp2016_for_fairness.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/vlsp2016_training.csv", split="train"
            )
            self.text = "Data"
            self.label = "Class"

        # Text Classification
        elif self.dataset_name == "PhoATIS":
            self.task = "text-classification_atis"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/PhoATIS.csv", split="train"
            )
            self.dataset_testing = self.dataset_testing.map(eval_keys("label"))
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/PhoATIS_training.csv", split="train"
            )
            self.dataset_training = self.dataset_training.map(
                eval_keys("label"))
            self.text = "sentence"
            self.label = "label"

        elif self.dataset_name == "PhoATIS_robustness":
            self.task = "text-classification_atis"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/PhoATIS_for_robustness.csv", split="train"
            )
            self.dataset_testing = self.dataset_testing.map(eval_keys("label"))
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/PhoATIS_training.csv", split="train"
            )
            self.dataset_training = self.dataset_training.map(
                eval_keys("label"))
            self.text = "sentence"
            self.label = "label"

        elif self.dataset_name == "PhoATIS_fairness":
            self.task = "text-classification_atis"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/PhoATIS_for_fairness.csv", split="train"
            )
            self.dataset_testing = self.dataset_testing.map(eval_keys("label"))
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/PhoATIS_training.csv", split="train"
            )
            self.dataset_training = self.dataset_training.map(
                eval_keys("label"))
            self.text = "sentence"
            self.label = "label"

        elif self.dataset_name == "UIT-VSMEC":
            self.task = "text-classification_vsmec"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/UIT-VSMEC.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/UIT-VSMEC_training.csv", split="train"
            )
            self.text = "Sentence"
            self.label = "Label"

        elif self.dataset_name == "UIT-VSMEC_robustness":
            self.task = "text-classification_vsmec"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/UIT-VSMEC_for_robustness.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/UIT-VSMEC_training.csv", split="train"
            )
            self.text = "Sentence"
            self.label = "Label"

        elif self.dataset_name == "UIT-VSMEC_fairness":
            self.task = "text-classification_vsmec"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/UIT-VSMEC_for_fairness.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/UIT-VSMEC_training.csv", split="train"
            )
            self.text = "Sentence"
            self.label = "Label"

        # Knowledge
        elif self.dataset_name == "zalo_e2eqa":
            self.task = "knowledge_openended"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/zalo_e2eqa.csv", split="train"
            )
            self.question = "question"
            self.answer = "answers"

        elif self.dataset_name == "zalo_e2eqa_robustness":
            self.task = "knowledge_openended"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/zalo_e2eqa_robustness.csv", split="train"
            )
            self.question = "question"
            self.answer = "answers"

        elif self.dataset_name == "ViMMRC":
            self.task = "knowledge_mtpchoice"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/ViMMRC.csv", split="train"
            )
            self.dataset_testing = self.dataset_testing.map(
                eval_keys("options"))
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/ViMMRC_training.csv", split="train"
            )
            self.dataset_training = self.dataset_training.map(
                eval_keys("options"))
            self.context = "article"
            self.question = "question"
            self.options = "options"
            self.answer = "answer"

        elif self.dataset_name == "ViMMRC_robustness":
            self.task = "knowledge_mtpchoice"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/ViMMRC_robustness.csv", split="train"
            )
            self.dataset_testing = self.dataset_testing.map(
                eval_keys("options"))
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/ViMMRC_training.csv", split="train"
            )
            self.dataset_training = self.dataset_training.map(
                eval_keys("options"))
            self.context = "article"
            self.question = "question"
            self.options = "options"
            self.answer = "answer"

        # Toxicity Detection
        elif self.dataset_name == "ViCTSD":
            self.task = "toxicity-detection"
            self.dataset_testing = load_dataset(
                "tarudesu/ViCTSD", split="test")
            self.dataset_training = load_dataset(
                "tarudesu/ViCTSD", split="train")
            self.text = "Comment"
            self.label = "Toxicity"

        elif self.dataset_name == "ViCTSD_robustness":
            self.task = "toxicity-detection"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/ViCTSD_for_robustness.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "tarudesu/ViCTSD", split="train")
            self.text = "Comment"
            self.label = "Toxicity"

        elif self.dataset_name == "ViCTSD_fairness":
            self.task = "toxicity-detection"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/ViCTSD_for_fairness.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "tarudesu/ViCTSD", split="train")
            self.text = "Comment"
            self.label = "Toxicity"

        elif self.dataset_name == "ViHSD":
            self.task = "toxicity-detection"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/ViHSD.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/ViHSD_training.csv", split="train"
            )
            self.text = "free_text"
            self.label = "label_id"

        elif self.dataset_name == "ViHSD_robustness":
            self.task = "toxicity-detection"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/ViHSD_for_robustness.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/ViHSD_training.csv", split="train"
            )
            self.text = "free_text"
            self.label = "label_id"

        elif self.dataset_name == "ViHSD_fairness":
            self.task = "toxicity-detection"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/ViHSD_for_fairness.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/ViHSD_training.csv", split="train"
            )
            self.text = "free_text"
            self.label = "label_id"

        # Information Retrieval
        elif self.dataset_name == "mmarco":
            raise NotImplementedError

        elif self.dataset_name == "mmarco_robustness":
            raise NotImplementedError

        elif self.dataset_name == "mmarco_fairness":
            raise NotImplementedError

        elif self.dataset_name == "mrobust":
            raise NotImplementedError

        elif self.dataset_name == "mrobust_robustness":
            raise NotImplementedError

        elif self.dataset_name == "mrobust_fairness":
            raise NotImplementedError

        # Language
        elif self.dataset_name == "mlqa_MLM":
            self.task = "language-modelling_filling"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/mlqa_MLM.csv", split="train"
            )
            self.context = "context"
            self.masked = "masked"

        elif self.dataset_name == "mlqa_MLM_fairness":
            self.task = "language-modelling_filling"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/mlqa_MLM_for_fairness.csv", split="train"
            )
            self.context = "context"
            self.masked = "masked"

        elif self.dataset_name == "VSEC":
            self.task = "language-modelling_correction"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/VSEC.csv", split="train"
            )
            self.source = "text"
            self.target = "correct"

        elif self.dataset_name == "VSEC_fairness":
            self.task = "language-modelling_correction"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/VSEC_for_fairness.csv", split="train"
            )
            self.source = "text"
            self.target = "correct"

        # Reasoning
        elif self.dataset_name == "synthetic_natural":
            self.task = "reasoning_synthetic"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/synthetic_reasoning_natural.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/synthetic_reasoning_natural.csv", split="train"
            )
            self.source = "question"
            self.target = "target"

        elif self.dataset_name == "synthetic_induction":
            self.task = "reasoning_synthetic"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/synthetic_reasoning_induction.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/synthetic_reasoning_induction.csv", split="train"
            )
            self.source = "source"
            self.target = "target"

        elif self.dataset_name == "synthetic_pattern":
            self.task = "reasoning_synthetic"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/synthetic_reasoning_pattern_match.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/synthetic_reasoning_pattern_match.csv", split="train"
            )
            self.source = "source"
            self.target = "target"

        elif self.dataset_name == "synthetic_induction":
            self.task = "reasoning_synthetic"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/synthetic_reasoning_variable_substitution.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/synthetic_reasoning_variable_substitution.csv", split="train"
            )
            self.source = "source"
            self.target = "target"

        elif self.dataset_name.startswith("math_level1"):
            self.task = "reasoning_math"
            subset = self.dataset_name.split("_")[-1]
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/math_level1.csv", split="train"
            )
            self.dataset_testing = self.dataset_testing.filter(
                lambda x: x["type"] == subset)
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/math_level1.csv", split="train"
            )
            self.dataset_training = self.dataset_training.filter(
                lambda x: x["type"] == subset)
            self.question = "problem"
            self.type = "type"
            self.answer = "solution"

        # Translation
        elif self.dataset_name == "PhoMT_envi":
            self.task = "translation_envi"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/PhoMT.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/PhoMT_training.csv", split="train"
            )
            self.source_language = "en"
            self.target_language = "vi"

        elif self.dataset_name == "PhoMT_vien":
            self.task = "translation_vien"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/PhoMT.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/PhoMT_training.csv", split="train"
            )
            self.source_language = "vi"
            self.target_language = "en"

        elif self.dataset_name == "PhoMT_envi_robustness":
            self.task = "translation_envi"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/PhoMT_for_robustness.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/PhoMT_training.csv", split="train"
            )
            self.source_language = "en"
            self.target_language = "vi"

        elif self.dataset_name == "PhoMT_vien_robustness":
            self.task = "translation_vien"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/PhoMT_for_robustness.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "csv", data_files="training_datasets/PhoMT_training.csv", split="train"
            )
            self.source_language = "vi"
            self.target_language = "en"

        elif self.dataset_name == "opus100_envi":
            self.task = "translation_envi"
            self.dataset_testing = load_dataset(
                "vietgpt/opus100_envi", split="test")
            self.dataset_training = load_dataset(
                "vietgpt/opus100_envi", split="train")
            self.source_language = "en"
            self.target_language = "vi"

        elif self.dataset_name == "opus100_vien":
            self.task = "translation_vien"
            self.dataset_testing = load_dataset(
                "vietgpt/opus100_envi", split="test")
            self.dataset_training = load_dataset(
                "vietgpt/opus100_envi", split="train")
            self.source_language = "vi"
            self.target_language = "en"

        elif self.dataset_name == "opus100_envi_robustness":
            self.task = "translation_envi"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/opus100_envi_for_robustness.csv", split="train"
            )
            self.dataset_training = load_dataset(
                "vietgpt/opus100_envi", split="train")
            self.source_language = "en"
            self.target_language = "vi"

        elif self.dataset_name == "opus100_vien_robustness":
            self.task = "translation_vien"
            self.dataset_testing = load_dataset(
                "csv", data_files="evaluation_datasets/opus100_envi_for_robustness.csv", split="train"
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
