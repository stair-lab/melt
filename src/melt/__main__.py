import spacy
import nltk

# Download the 'punkt' tokenizer models from NLTK
nltk.download('punkt')

# Try to load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print(
        "Downloading the spaCy en_core_web_sm model\n"
        "(don't worry, this will only happen once)"
    )
    from spacy.cli import download
    download("en_core_web_sm")
    # Reload the model after downloading
    nlp = spacy.load("en_core_web_sm")

# Import and execute the main function from cli module
# Adjust the import if this script is not part of a package
try:
    from cli import main  # Use relative import if part of a package
except ImportError:
    import cli
    main = cli.main

main()
