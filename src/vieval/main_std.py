import argparse
import os
from .metrics.text_classification import TextClassificationMetric
from .metrics.calibration_metric import CalibrationMetric
from .metrics.language import LanguageMetric
from .metrics.ir import InformationRetrievalMetric
from .metrics.translation_metric import TranslationMetric
from .metrics.question_answering import QAMetric
from .metrics.reasoning import ReasoningMetric
from .metrics.summary import SummaryMetric
from .metrics.bias import BiasMetric
from .metrics.toxicity import ToxicityMetric
import numpy as np
from .metrics.utils import read_json_file, save_to_json
from time import time
import logging
import traceback
from typing import List, Dict


def get_std(result_list: List) -> Dict:
    temp = {}
    final_result = {}
    for result in result_list:
        for k in result.keys():
            if result[k]:
                temp[k] = temp[k] + [result[k]] if k in temp else [result[k]]
    
    # print(temp)
    for k in temp.keys():
        if temp[k]:
            final_result[f"{k}_std"] = np.array(temp[k]).std()
            
    return final_result


def get_subdata(data: Dict, n: int, indices) -> Dict:
    sub_data = {}
    for key in data.keys():
        if isinstance(data[key], list) and len(data[key]) == n:
            sub_data[key] = [data[key][i] for i in indices]
            print(key, len(sub_data[key]))
        else:
            sub_data[key] = data[key]
    
    return sub_data


def evaluation(args):
    args.key_answer = None
    args.class_names = None
    args.mode = "fewshot" if "fewshot" in args.filepath else "zeroshot"
    if "vietnews" in args.filepath:
        metric = SummaryMetric()
    elif "wiki_lingua" in args.filepath:
        metric = SummaryMetric()
    elif "ViMMRC" in args.filepath:
        args.class_names = ["A", "B", "C", "D"]
        args.key_answer = "choice"
        metric = TextClassificationMetric()
    elif "ViHSD" in args.filepath:
        args.key_answer = "toxic_level"
        args.class_names = [0, 1, 2]
        metric = TextClassificationMetric()
    elif "PhoATIS" in args.filepath:
        args.class_names = [i for i in range(17)]
        args.key_answer = "tag"
        metric = TextClassificationMetric()
    elif "mlqa_MLM" in args.filepath:
        metric = LanguageMetric()
    elif "mmarco" in args.filepath:
        args.key_answer = "answer"
        args.class_names = ["yes", "no"]
        metric = InformationRetrievalMetric()
    elif "mrobust" in args.filepath:
        args.key_answer = "answer"
        args.class_names = ["yes", "no"]
        metric = InformationRetrievalMetric()
    elif "PhoMT" in args.filepath:
        args.key_answer = "translation"
        metric = TranslationMetric()
    elif "opus" in args.filepath:
        args.key_answer = "translation"
        metric = TranslationMetric()
    elif "srabstract" in args.filepath:
        args.key_answer = "answer"
        metric = ReasoningMetric()
    elif "srnatural" in args.filepath:
        args.key_answer = "answer"
        metric = ReasoningMetric()
    elif "math" in args.filepath:
        args.key_answer = "answer"
        args.keep_punc = True
        metric = ReasoningMetric()
    elif "UIT-VSFC" in args.filepath:
        args.key_answer = "sentiment"
        args.class_names = [0, 1, 2]
        metric = TextClassificationMetric()
    elif "UIT-VSMEC" in args.filepath:
        args.key_answer = "emotion"
        args.class_names = [0, 1, 2, 3, 4, 5, 6]
        metric = TextClassificationMetric()
    elif "ViCTSD" in args.filepath:
        args.key_answer = "toxic_level"
        args.class_names = [0, 1]
        metric = TextClassificationMetric()
    elif "vlsp2016" in args.filepath:
        args.key_answer = "sentiment"
        args.class_names = [0, 1, 2]
        metric = TextClassificationMetric()
    elif "VSEC" in args.filepath:
        metric = LanguageMetric()
    elif "zalo_e2eqa" in args.filepath:
        args.key_answer = "answer"
        metric = QAMetric()
    elif "mlqa" in args.filepath:
        metric = QAMetric()
    elif "xquad_xtreme" in args.filepath:
        metric = QAMetric()
    else:
        print(f"Not implement metric for {args.filepath} yet")
        return

    print(f"Read data from {args.filepath}")
    begin = time()
    data = read_json_file(filepath=args.filepath)
    print(f"Read data took {time()-begin:2f}")
    n = len(data['predictions'])
    results_lst = []
    for i in range(args.n_bootstrap):
        indices = np.random.choice(np.arange(n), size = int(args.p_bootstrap * n), replace = True)
        print(n, len(indices))
        sub_data = get_subdata(data, n, indices)
        result = {}
        print("Run accuracy metrics")
        begin = time()
        _, result = metric.evaluate(data=sub_data, args=args)
        print(f"Run accuracy took {time()-begin:2f}")

        if type(metric) is TextClassificationMetric:
            print("Run calibration metrics")
            begin = time()
            calibration_metric = CalibrationMetric()
            _, cal_result = calibration_metric.evaluate(data=sub_data, args=args)
            result.update(cal_result)
            print(f"Run calibration took {time()-begin:2f}")

        if type(metric) not in [TextClassificationMetric,
                                InformationRetrievalMetric,
                                ReasoningMetric]:
            print("Run bias metrics")
            begin = time()
            bias_metric = BiasMetric(data, args)
            for demographic_category in ['race', 'gender']:
                for target_category in ['profession']:
                    args.demographic_category = demographic_category
                    args.target_category = target_category
                    _, bias_result = bias_metric.evaluate(data=sub_data, args=args)
                    result.update(bias_result)
            print(f"Runni bias metrics took {time()-begin:2f}")

        if type(metric) not in [TextClassificationMetric,
                                LanguageMetric,
                                InformationRetrievalMetric,
                                ReasoningMetric]:
            print("Run toxicity metrics")
            begin = time()
            toxicity_metric = ToxicityMetric()
            _, toxic_result = toxicity_metric.evaluate(data=sub_data, args=args)
            result.update(toxic_result)
            print(f"Run toxicity metrics took {time()-begin:2f}")
        
        print(result)
        results_lst.append(result)
    
    result_final = get_std(results_lst)
    print(result_final)
    data['data'] = results_lst
    args.filename = os.path.basename(args.filepath)
    os.makedirs(args.out_eval_dir, exist_ok=True)
    save_to_json(data=data,
                 filename=f"specific_{args.filename}",
                 outdir=args.out_eval_dir)
    save_to_json(data=result_final,
                 filename=f"overall_{args.filename}",
                 outdir=args.out_eval_dir)


def std_estimation(args):
    np.random.seed(args.seed)
    os.makedirs(args.out_eval_dir, exist_ok=True)
    generation_files = [
        f for f in os.listdir(args.output_dir) if f.endswith(".json")
    ]
    generation_filepaths = [
        os.path.join(args.output_dir, f) for f in generation_files
    ]

    for filepath in generation_filepaths:
        try:
            logging.info(f"Runing {filepath}")
            args.filepath = filepath
            evaluation(args)
        except Exception as e:
            traceback.print_exc()
            logging.error(f"Error in {filepath}")
            logging.error(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output_dir", default="", type=str, help=""
    )
    parser.add_argument(
        "--out_eval_dir",
        default="./out_new",
        type=str,
        help="The output folder to save bias score",
    )
    parser.add_argument("--device", default="cuda:0", type=str, help="")
    parser.add_argument("--n_bootstrap", default=1000, type=int, help="")
    parser.add_argument("--p_bootstrap", default=0.9, type=int, help="")
    parser.add_argument("--seed", default=42, type=int, help="for bias metrics")
    parser.add_argument("--bs", default=128, type=int, help="for bias metrics")

    args = parser.parse_args()
    main(args)
