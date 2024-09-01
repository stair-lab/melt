import spacy
from spacy.cli import download
from transformers import HfArgumentParser
from dotenv import load_dotenv

from script_arguments import ScriptArguments  # Ensure this module is in the correct path
from generation import generation  # Ensure this module is in the correct path

def ensure_spacy_model(model_name="en_core_web_sm"):
    """
    Ensure the spaCy model is available. Download it if not present.
    """
    try:
        spacy.load(model_name)
        print(f"spaCy model '{model_name}' is already installed.")
    except OSError:
        print(f"spaCy model '{model_name}' not found. Downloading...")
        download(model_name)
        print(f"spaCy model '{model_name}' has been downloaded and installed.")

def main():
    """
    Main function to:
    1. Load environment variables from a .env file.
    2. Ensure the spaCy model is available.
    3. Parse command-line arguments.
    4. Execute the generation function with the parsed arguments.
    """
    # Load environment variables
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
