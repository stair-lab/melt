"""
This module provides basic metrics for evaluating text similarity and overlap.

It includes functions for exact match and F1 score calculations between
predicted text and gold standard text.
"""

from .utils import normalize_text

try:
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError as e:
    print(f"Error importing NLTK: {e}")
    # Handle the error or raise an exception

def exact_match(gold: str, pred: str) -> float:
    """Calculates whether the predicted text (pred)
    exactly matches the gold standard text (gold)
    after both texts have been normalized.

    Args:
        gold (str): The reference text that is considered
        the correct or expected result.
        pred (str): The text produced by a predictive model or
        some process that is being evaluated against the gold standard.

    Returns:
        float: The function returns a float, which will be 1.0
        if the normalized pred string exactly matches
        the normalized gold string, and 0.0 otherwise.
    """
    if not gold or not pred:
        return 0.0

    return 1.0 if normalize_text(gold) == normalize_text(pred) else 0.0

def f1_score(gold: str, pred: str) -> float:
    """Computes the F1 score for the overlap between
    the predicted text (pred) and the gold standard text (gold).

    Args:
        gold (str): The reference text that is
        considered the correct or expected result.
        pred (str): The text produced by a predictive model
        or some process that is being evaluated against the gold standard.

    Returns:
        float: The F1 score, ranging from 0.0 to 1.0, where 0.0 indicates
        no overlap and 1.0 indicates perfect overlap between gold and pred.
    """
    if not gold or not pred:
        return 0.0

    gold_tokens = set(word_tokenize(normalize_text(gold)))
    pred_tokens = set(word_tokenize(normalize_text(pred)))

    if not gold_tokens and not pred_tokens:
        return 1.0

    intersection = gold_tokens.intersection(pred_tokens)
    if not intersection:
        return 0.0
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
