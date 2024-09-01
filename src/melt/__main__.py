import logging
import spacy
import nltk
from spacy.cli import download as spacy_download
from typing import NoReturn

# Configure logging with a descriptive name for the logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("nlp_utils")


def download_nltk_resources() -> NoReturn:
    """Download the necessary NLTK resources.

    Logs success or failure messages.
    """
    try:
        with nltk.download('punkt'):
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


def execute_cli_main() -> None:
    """Execute the 'main' function from the CLI module.

    Logs success or failure messages about the import process and execution.
    """
    try:
        from cli import main as cli_main
        logger.info("Successfully imported 'main' from 'cli' module.")
    except ImportError as import_error:
        logger.error("ImportError: %s", import_error)
        try:
            import cli
            cli_main = cli.main
            logger.info("Successfully imported 'cli' module directly.")
        except ImportError as inner_import_error:
            logger.critical("Failed to import 'cli' module: %s", inner_import_error)
            raise
    cli_main()


def main() -> None:
    """Main function to set up resources and execute the CLI.

    Ensures proper logging and execution flow.
    """
    download_nltk_resources()
    load_spacy_model()
    execute_cli_main()


if __name__ == "__main__":
    main()
