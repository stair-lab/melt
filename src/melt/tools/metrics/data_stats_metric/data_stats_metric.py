"""
This module provides the DataStatsMetric class for evaluating coverage, density, and compression
of summaries based on tokenized input text.
"""

from collections import Counter
from multiprocessing import Pool
import subprocess
import sys
import pkg_resources

# Import statements
try:
    import gin
except ImportError:
    print("gin-config package is not installed.")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gin-config'])
    import gin

try:
    import spacy
    from spacy.cli import download
except ImportError:
    print("spacy package is not installed.")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'spacy'])
    import spacy
    from spacy.cli import download

from ..utils import Fragments

# Ensure required packages are installed
def install_packages():
    """
    Check for and install required packages if they are missing.
    """
    required_packages = ['gin-config', 'spacy']
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    missing_packages = [pkg for pkg in required_packages if pkg not in installed_packages]

    if missing_packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_packages])

install_packages()

# Load spacy model
try:
    _en = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    _en = spacy.load("en_core_web_sm")

def find_ngrams(input_list, n):
    """Return n-grams from input list."""
    return zip(*[input_list[i:] for i in range(n)])

@gin.configurable
class DataStatsMetric:
    """Class for calculating data statistics on text."""

    def __init__(self, n_gram=3, n_workers=24, case=False, tokenize=True):
        self.n_gram = n_gram
        self.n_workers = n_workers
        self.case = case
        self.tokenize = tokenize

    def evaluate_example(self, summary, input_text):
        """Evaluate a single summary against input text."""
        if self.tokenize:
            input_text, summary = self.tokenize_text(input_text, summary)

        fragments = Fragments(summary, input_text, case=self.case)
        score_dict = self.calculate_scores(fragments)

        for i in range(1, self.n_gram + 1):
            self.calculate_ngram_scores(fragments, i, score_dict)

        return score_dict

    def tokenize_text(self, input_text, summary):
        """Tokenize the input text and summary."""
        input_text = _en(input_text, disable=["tagger", "parser", "ner", "textcat"])
        input_text = [tok.text for tok in input_text]
        summary = _en(summary, disable=["tagger", "parser", "ner", "textcat"])
        summary = [tok.text for tok in summary]
        return input_text, summary

    def calculate_scores(self, fragments):
        """Calculate coverage, density, and compression scores."""
        coverage = fragments.coverage()
        density = fragments.density()
        compression = fragments.compression()
        tokenized_summary = fragments.get_summary()  # Ensure Fragments has this method
        return {
            "coverage": coverage,
            "density": density,
            "compression": compression,
            "summary_length": len(tokenized_summary),
        }

    def calculate_ngram_scores(self, fragments, n, score_dict):
        """Calculate n-gram related scores."""
        tokenized_summary = fragments.get_summary()  # Ensure Fragments has this method
        tokenized_text = fragments.get_text()  # Ensure Fragments has this method

        input_ngrams = list(find_ngrams(tokenized_text, n))
        summ_ngrams = list(find_ngrams(tokenized_summary, n))
        input_ngrams_set = set(input_ngrams)
        summ_ngrams_set = set(summ_ngrams)
        intersect = summ_ngrams_set.intersection(input_ngrams_set)

        if len(summ_ngrams_set) > 0:
            score_dict[f"percentage_novel_{n}-gram"] = (
                len(summ_ngrams_set) - len(intersect)
            ) / float(len(summ_ngrams_set))
            ngram_counter = Counter(summ_ngrams)
            repeated = [key for key, val in ngram_counter.items() if val > 1]
            score_dict[f"percentage_repeated_{n}-gram_in_summ"] = (
                len(repeated) / float(len(summ_ngrams_set))
            )
        else:
            score_dict[f"percentage_novel_{n}-gram"] = 0.0
            score_dict[f"percentage_repeated_{n}-gram_in_summ"] = 0.0

    def evaluate_batch(self, summaries, input_texts, aggregate=True):
        """Evaluate multiple summaries against input texts."""
        corpus_score_dict = Counter()
        with Pool(processes=self.n_workers) as p:
            results = p.starmap(self.evaluate_example, zip(summaries, input_texts))

        if aggregate:
            for result in results:
                corpus_score_dict.update(result)
            if len(input_texts) > 0:
                for key in corpus_score_dict.keys():
                    corpus_score_dict[key] /= float(len(input_texts))
            return corpus_score_dict
        return results

    @property
    def supports_multi_ref(self):
        """Check if multiple references are supported."""
        return False
