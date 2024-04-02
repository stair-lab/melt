from .script_arguments import ScriptArguments
from .generation import generation
from .main_mean import mean_estimation
from .main_std import std_estimation
from .to_sheet import to_sheet
from .to_sheet_std import to_sheet_std
from transformers import HfArgumentParser
from dotenv import load_dotenv


def main():
    load_dotenv()
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    if args.mode == "generation":
        generation(args)
    elif args.mode == "evaluation":
        mean_estimation(args)
        std_estimation(args)
        to_sheet(args)
        to_sheet_std(args)
    elif args.mode == "end2end":
        generation(args)
        mean_estimation(args)
        std_estimation(args)
        to_sheet(args)
        to_sheet_std(args)
    else:
        print("ERROR: No such mode '{}'".format(args.mode))
