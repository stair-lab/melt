from .post_process import get_answer_auto_from_text


class BaseMetric:
    """An abstract class that provides a foundation for various metric classes to evaluate different aspects of text data.
    """
    def __init__(self):
        return

    def _get_answer(self, text: str, args) -> str:
        """Process a text and extract an answer based on certain arguments

        Args:
            text (str): A string containing the text from which the answer is to be extracted.
        """
        return get_answer_auto_from_text(
            text=text,
            key_answer=args.key_answer,
            class_names=args.class_names,
            args=args,
        )
