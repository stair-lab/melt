from .script_arguments import ScriptArguments
from .generation import generation
from .evaluation import evaluation
# from .to_sheet import to_sheet
# from .to_sheet_std import to_sheet_std
from transformers import HfArgumentParser
from dotenv import load_dotenv


def main():
    load_dotenv()
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    if args.mode == "generation":
        generation(args)
    elif args.mode == "evaluation":
        evaluation(args)
    
    elif args.mode == "end2end":
        generation(args)
        evaluation(args)
        
    else:
        raise ValueError("ERROR: No such mode '{}'".format(args.mode))
