# pylint: disable=C0103,W0221,W0106
from collections import Counter
from multiprocessing import Pool
import gin
import spacy
from ..utils import Fragments


try:
    _en = spacy.load("en_core_web_sm")
except OSError:
    print(
        "Downloading the spacy en_core_web_sm model\n"
        "(don't worry, this will only happen once)"
    )
    from spacy.cli import download

    download("en_core_web_sm")
    _en = spacy.load("en_core_web_sm")


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


@gin.configurable
class DataStatsMetric:
    def __init__(self, n_gram=3, n_workers=24, case=False, tokenize=True):
        self.n_gram = n_gram
        self.n_workers = n_workers
        self.case = case
        self.tokenize = tokenize

    def evaluate_example(self, summary, input_text):
        if self.tokenize:
            input_text = _en(
                input_text, disable=["tagger", "parser", "ner", "textcat"]
            )
            input_text = [tok.text for tok in input_text]
            summary = _en(
                summary, disable=["tagger", "parser", "ner", "textcat"]
            )
            summary = [tok.text for tok in summary]
        fragments = Fragments(summary, input_text, case=self.case)
        coverage = fragments.coverage()
        density = fragments.density()
        compression = fragments.compression()
        score_dict = {
            "coverage": coverage,
            "density": density,
            "compression": compression,
        }
        tokenized_summary = fragments._norm_summary
        tokenized_text = fragments._norm_text
        score_dict["summary_length"] = len(tokenized_summary)
        for i in range(1, self.n_gram + 1):
            input_ngrams = list(find_ngrams(tokenized_text, i))
            summ_ngrams = list(find_ngrams(tokenized_summary, i))
            input_ngrams_set = set(input_ngrams)
            summ_ngrams_set = set(summ_ngrams)
            intersect = summ_ngrams_set.intersection(input_ngrams_set)
            try:
                score_dict[f"percentage_novel_{i}-gram"] = (
                    len(summ_ngrams_set) - len(intersect)
                ) / float(len(summ_ngrams_set))
                ngramCounter = Counter()
                ngramCounter.update(summ_ngrams)
                repeated = [
                    key for key, val in ngramCounter.items() if val > 1
                ]
                score_dict[f"percentage_repeated_{i}-gram_in_summ"] = len(
                    repeated
                ) / float(len(summ_ngrams_set))
            except ZeroDivisionError:
                continue
        return score_dict

    def evaluate_batch(self, summaries, input_texts, aggregate=True):
        corpus_score_dict = Counter()
        p = Pool(processes=self.n_workers)
        results = p.starmap(self.evaluate_example, zip(summaries, input_texts))
        p.close()
        if aggregate:
            [corpus_score_dict.update(x) for x in results]
            for key in corpus_score_dict.keys():
                corpus_score_dict[key] /= float(len(input_texts))
            return corpus_score_dict
        else:
            return results

    @property
    def supports_multi_ref(self):
        return False
