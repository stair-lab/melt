"""
This module contains base classes for metrics processing.
"""

from melt.tools.metrics.post_process import get_answer_auto_from_text

class BaseMetric:
    """
    A base class for metrics that process text and extract answers.
    """

    def __init__(self, data=None, args=None):
        """
        Initializes the BaseMetric with optional data and arguments.
        """
        self.data = data
        self.args = args

    def _get_answer(self, text: str, args) -> str:
        """
        Process a text and extract an answer based on certain arguments.
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
        """
        self.data = data

    def get_data(self):
        """
        Gets the data for the metric.
        """
        return self.data
