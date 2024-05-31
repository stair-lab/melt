from .utils import f_measure, normalize_text


def exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return 1 if normalize_text(gold) == normalize_text(pred) else 0


def f1_score(gold: str, pred: str) -> float:
    ret = f_measure(
        set(normalize_text(gold).split()), set(normalize_text(pred).split())
    )
    if ret is None:  # answer is the empty string after normalizing
        return 0.0

    return ret
