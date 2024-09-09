"""Base wrapper module for text processing tools.

This module provides an abstract base class for text processing wrappers.
Subclasses should implement the methods defined in this base class.
"""

class BaseWrapper:
    """Abstract base class for text processing wrappers.

    This class defines the interface that all text processing wrappers should
    implement. It includes methods for handling prompts and computing log
    probabilities.
    """
    def __call__(self, prompts, return_probs=False):
        """Process the given prompts.

        Args:
            prompts (list of str): The prompts to be processed.
            return_probs (bool, optional): Whether to return probabilities. Defaults to False.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def compute_logprob_and_length(self, prompts, completions):
        """Compute the log probabilities and lengths for given prompts and completions.

        Args:
            prompts (list of str): The prompts to evaluate.
            completions (list of str): The completions to evaluate.

        Returns:
            tuple: A tuple containing the log probabilities and lengths.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError
