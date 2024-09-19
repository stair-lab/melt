"""Base wrapper module for text processing tools"""

class BaseWrapper:
    """Abstract base class for text processing wrappers"""
    def __call__(self, prompts, return_probs=False):
        """Process the given prompts."""
        raise NotImplementedError

    def compute_logprob_and_length(self, prompts, completions):
        """Compute the log probabilities and lengths for given prompts and completions."""
        raise NotImplementedError
