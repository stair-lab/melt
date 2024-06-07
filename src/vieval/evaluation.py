import traceback
from typing import List, Dict, Any
import os
import argparse
import logging
from utils.metric_utils import read_json_file, save_to_json, info_from_filename




def metric_main(args):
    data = read_json_file(filepath=args.filepath)
    pipeline = MetricPipeline()
    filename = os.path.basename(args.filepath)
    task_name, ds_name, _, _ , _ = info_from_filename(filename)
    mean_results = pipeline.run_mean(data, task_name, ds_name, args)
    save_to_json(mean_results, filename, args.out_eval_dir)
    

def evaluation(args):
    os.makedirs(args.output_eval_dir, exist_ok=True)
    
    if args.filepath:
        try:
            metric_main(args)
        except Exception as e:
            traceback.print_exc()
            logging.error(f"Error in {args.filepath}")
            logging.error(e)
    else:
        generation_files = [
            f for f in os.listdir(args.output_dir) if f.endswith(".json")
        ]
        generation_filepaths = [
            os.path.join(args.output_dir, f) for f in generation_files
        ]

        for filepath in generation_filepaths:
            if args.pattern is None or args.pattern in filepath:
                try:
                    logging.info(f"Runing {filepath}")
                    args.filepath = filepath
                    metric_main(args)
                except Exception as e:
                    traceback.print_exc()
                    logging.error(f"Error in {filepath}")
                    logging.error(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--generation_dir", default="", type=str, help=""
    )
    parser.add_argument(
        "--out_dir",
        default="./out_new",
        type=str,
        help="The output folder to save bias score",
    )
    parser.add_argument(
        "--filepath", default=None, type=str, help=""
    )
    parser.add_argument("--device", default="cuda:0", type=str, help="")
    parser.add_argument("--n_bootstrap", default=2, type=int, help="")
    parser.add_argument("--p_bootstrap", default=1.0, type=int, help="")
    parser.add_argument("--seed", default=42, type=int, help="for bias metrics")
    parser.add_argument("--bs", default=128, type=int, help="for bias metrics")
    parser.add_argument("--pattern", default=None, type=str, help="")

    args = parser.parse_args()

    logging.basicConfig(
        filename="app.log",
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main(args)
