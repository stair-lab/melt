"""
This script initializes NLP models and runs the main function from the 'cli' module.

The script performs the following tasks:
1. Downloads the 'punkt' tokenizer models using nltk.
2. Loads the spaCy 'en_core_web_sm' model, downloading it if necessary.
3. Imports and executes the 'main' function from the 'cli' module.

If any module or function cannot be imported, appropriate error messages are displayed.
"""

import logging
import cli
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("nlp_utils")
try:
    import spacy
    logger.info("Successfully imported 'spacy' module.")
    # You can include other code that uses spacy here
except ImportError as import_error:
    logger.error("Failed to import 'spacy': %s", import_error)
    # Handle the import failure (e.g., exit the program or take alternative actions)
    raise  # Optionally, re-raise the exception if you want to stop execution
try:
    import nltk
    logger.info("Successfully imported 'nltk' module.")
    # You can include other code that uses nltk here
except ImportError as import_error:
    logger.error("Failed to import 'nltk': %s", import_error)
    # Handle the import failure (e.g., exit the program or take alternative actions)
    raise  # Optionally, re-raise the exception if you want to stop execution

try:
    from spacy.cli import download as spacy_download
    logger.info("Successfully imported 'spacy.cli.download' as 'spacy_download'.")
    # You can include code that uses spacy_download here
except ImportError as import_error:
    logger.error("Failed to import 'spacy.cli.download': %s", import_error)
    # Handle the import failure (e.g., exit the program or take alternative actions)
    raise  # Optionally, re-raise the exception if you want to stop execution

# Configure logging with a descriptive name for the logger


def execute_cli_main() -> None:
    """Execute the 'main' function from the CLI module.

    Logs success or failure messages about the import process and execution.
    """
    try:
        cli_main = cli.main
        logger.info("Successfully imported 'main' from 'cli' module.")
    except AttributeError as attr_error:
        logger.error("AttributeError: %s", attr_error)
        logger.critical("Failed to find 'main' function in 'cli' module.")
        raise
    try:
        cli_main()
    except Exception as e:
        logger.error("Failed to execute 'cli_main': %s", e)
        raise

def download_nltk_resources() -> None:
    """Download the necessary NLTK resources.

    Logs success or failure messages.
    """
    try:
        nltk.download('punkt')
        logger.info("Successfully downloaded NLTK 'punkt' resource.")
    except Exception as error:
        logger.error("Failed to download NLTK resources: %s", error)
        raise

def load_spacy_model(model_name: str = "en_core_web_sm") -> spacy.language.Language:
    """Load and return the spaCy model, downloading it if necessary.

    Logs success or failure messages during the model loading process.

    Args:
        model_name (str): The name of the spaCy model to load.

    Returns:
        spacy.language.Language: The loaded spaCy model.
    """
    try:
        model = spacy.load(model_name)
        logger.info("Successfully loaded spaCy model: %s", model_name)
    except OSError:
        logger.warning("spaCy model '%s' not found. Downloading...", model_name)
        spacy_download(model_name)
        model = spacy.load(model_name)
        logger.info("Successfully downloaded and loaded spaCy model: %s", model_name)
    except Exception as error:
        logger.error("Failed to load spaCy model: %s", error)
        raise
    return model

def main() -> None:
    """Main function to set up resources and execute the CLI.

    Ensures proper logging and execution flow.
    """
    try:
        download_nltk_resources()
        logger.info("Successfully downloaded NLTK resources.")
    except (nltk.NLPException, FileNotFoundError) as e:
        logger.error("Failed to download NLTK resources: %s", e)
        return  # or raise to propagate the error

    try:
        load_spacy_model()
        logger.info("Successfully loaded spaCy model.")
    except (spacy.errors.SpacyException, ImportError) as e:
        logger.error("Failed to load spaCy model: %s", e)
        return  # or raise to propagate the error

    try:
        execute_cli_main()
    except Exception as e:
        logger.error("Failed to execute CLI main: %s", e)
        raise  # Reraise the exception to handle it at a higher level

if __name__ == "__main__":
    main()