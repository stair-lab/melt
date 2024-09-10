"""
This module contains base classes for metrics processing.
"""

from .post_process import get_answer_auto_from_text

class BaseMetric:
    """
    A base class for metrics that process text and extract answers.
    """

    def __init__(self, data=None, args=None):
        """
        Initializes the BaseMetric with optional data and arguments.

        Args:
            data (optional): Data related to the metric. Defaults to None.
            args (optional): Arguments for processing. Defaults to None.
        """
        self.data = data
        self.args = args

    def _get_answer(self, text: str, args) -> str:
        """
        Process a text and extract an answer based on certain arguments.

        Args:
            text (str): A string containing the text from which the answer is \
                to be extracted.
            args: Arguments containing 'key_answer', 'class_names', and other \
                parameters required for extraction.

        Returns:
            str: The extracted answer.
        """
        return get_answer_auto_from_text(
            text=text,
            key_answer=args.key_answer,
            class_names=args.class_names,
            args=args,
        )

    def set_data(self, data):
        """
        Sets the data for the metric.

        Args:
            data: The data to be set.
        """
        self.data = data

    def get_data(self):
        """
        Gets the data for the metric.

        Returns:
            The current data.
        """
        return self.data
