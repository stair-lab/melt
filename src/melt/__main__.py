"""
This script initializes NLP models and runs the main function from the 'cli' module.

The script performs the following tasks:
1. Downloads the 'punkt' tokenizer models using nltk.
2. Loads the spaCy 'en_core_web_sm' model, downloading it if necessary.
3. Imports and executes the 'main' function from the 'cli' module.

If any module or function cannot be imported, appropriate error messages are displayed.
"""

try:
    import nltk
except ImportError as import_error:
    print(f"Error importing nltk: {import_error}")
else:
    try:
        # Attempt to download the 'punkt' tokenizer models
        nltk.download('punkt')
        print("nltk 'punkt' models downloaded successfully.")
    except ImportError:
        print(f"Error downloading nltk 'punkt' models: {ImportError}")

nltk.download('punkt')


try:
    import spacy
except ImportError as import_error:
    print(f"Error importing spacy: {import_error}")
else:
    try:
        # Attempt to load the spaCy model
        nlp = spacy.load("en_core_web_sm")
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
    except IOError as e:
        print(f"IO error occurred: {e}")
    except ValueError as e:
        print(f"Invalid model value: {e}")
    except ImportError:
        print(f"An unexpected error occurred while loading the spaCy model: {e}")
