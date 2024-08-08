from .utils import normalize_text
from nltk.metrics.scores import f_measure


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
    if not pred:
        return 0

    return 1 if normalize_text(gold) == normalize_text(pred) else 0


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
    ret = f_measure(
        set(normalize_text(gold).split()), set(normalize_text(pred).split())
    )
    if ret is None:  # answer is the empty string after normalizing
        return 0.0

    return ret
