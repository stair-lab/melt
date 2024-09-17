"""
This script initializes and runs the text generation pipeline using spaCy, 
transformers, and dotenv. It also handles downloading the spaCy 'en_core_web_sm' 
model if it is not already present.

The main function is responsible for:
1. Loading environment variables.
2. Parsing script arguments.
3. Running the generation process with the parsed arguments.
"""
try:
    import spacy
except ImportError as e:
    print(f"Failed to import 'spacy': {e}")

try:
    spacy.load("en_core_web_sm")
except OSError:
    print(
        "Downloading the spacy en_core_web_sm model\n"
        "(don't worry, this will only happen once)"
    )
    try:
        from spacy.cli import download
        download("en_core_web_sm")

    except ImportError as e:
        print(f"Failed to import 'spacy.cli': {e}")
try:
    from transformers import HfArgumentParser
except ImportError as e:
    print(f"Failed to import 'transformers': {e}")

try:
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Failed to import 'dotenv': {e}")

try:
    from .script_arguments import ScriptArguments
except ImportError as e:
    print(f"Failed to import 'ScriptArguments' from 'script_arguments': {e}")
try:
    from .generation import generation
except ImportError as e:
    print(f"Failed to import 'generation' from 'generation': {e}")

def main():
    """
    The main function that initializes the environment, parses script arguments,
    and triggers the text generation process.

    This function performs the following steps:
    1. Loads environment variables using `load_dotenv()`.
    2. Creates an argument parser for `ScriptArguments` using `HfArgumentParser`.
    3. Parses the arguments into data classes.
    4. Calls the `generation` function with the parsed arguments to perform the text generation.

    Returns:
        None
    """
    load_dotenv()

    # Ensure spaCy model is available
    ensure_spacy_model()

    # Parse command-line arguments
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Execute the generation function with parsed arguments
    generation(args)

if __name__ == "__main__":
    main()
