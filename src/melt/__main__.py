"Main"
import os
import sys
import spacy
import nltk
from melt.cli import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
nltk.download('punkt_tab')
try:
    spacy.load("en_core_web_sm")
except OSError:
    print(
        "Downloading the spacy en_core_web_sm model\n"
        "(don't worry, this will only happen once)"
    )
    from spacy.cli import download

    download("en_core_web_sm")

main()
