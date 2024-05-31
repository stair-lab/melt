class BaseWrapper:

    def __call__(self, prompts, return_probs=False):
        raise NotImplementedError

    def compute_logprob_and_length(self, prompts, completions):
        raise NotImplementedError
