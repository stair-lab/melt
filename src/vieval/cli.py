from .script_arguments import ScriptArguments
from .generation import generation
from transformers import HfArgumentParser
from dotenv import load_dotenv

def main():
    load_dotenv()
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    if args.mode == "generation":
        generation(args)
    else:
        pass