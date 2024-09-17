"""
This module provides the OpenAIWrapper class for interacting with OpenAI's
API with retry capabilities and custom configurations.
"""
try:
    import openai
except ModuleNotFoundError as e:
    print(f"Module not found: {e}")
except ImportError as e:
    print(f"Import error occurred: {e}")


try:
    import backoff
except ModuleNotFoundError as e:
    print(f"Module not found: {e}")
except ImportError as e:
    print(f"Import error occurred: {e}")



from .BaseWrapper import BaseWrapper


class OpenAIWrapper(BaseWrapper):
    """
    A wrapper class for interacting with the OpenAI API with retry capabilities.

    Attributes:
        generation_config (dict): Configuration for text generation.
        model (openai.OpenAI): OpenAI API model instance.
        engine (str): The engine to use for generating text.
    """
    def __init__(self, engine=None, generation_config=None):
        generation_config["max_tokens"] = generation_config.pop(
            "max_new_tokens"
        )
        generation_config["frequency_penalty"] = generation_config.pop(
            "repetition_penalty"
        )
        self.generation_config = generation_config
        self.model = openai.OpenAI()
        self.engine = engine

    def __call__(self, prompts, return_probs=False):
        """
        Generates text completions for a list of prompts.

        Args:
            prompts (list): A list of prompts to generate completions for.
            return_probs (bool): Whether to return probabilities (not implemented).

        Returns:
            tuple: A tuple containing lists of generations, probabilities, and token counts.
        """
        generations = []
        generations_probs = [[]] * len(prompts)
        num_generated_tokens = []
        for prompt in prompts:

            response = self.chat_completions_with_backoff(
                model=self.engine,
                messages=prompt,
                **self.generation_config,
            )

            generations.append(response.choices[0].message.content)
            num_generated_tokens.append(response.usage.completion_tokens)

        return generations, generations_probs, num_generated_tokens

    def compute_logprob_and_length(self, prompts, completions):
        """
        Computes log probabilities and lengths of completions.

        Args:
            prompts (list): A list of prompts.
            completions (list): A list of completions for the prompts.

        Returns:
            tuple: A tuple containing lists of log probabilities and token counts.
        """
        completions_num_tokens = [0] * len(prompts)
        completions_logprobs = [[]] * len(prompts)
        return completions_logprobs, completions_num_tokens

    @backoff.on_exception(backoff.expo, openai.OpenAIError, max_tries=10)
    def chat_completions_with_backoff(self, **kwargs):
        """
        Retrieves chat completions from the OpenAI API with retry capabilities.

        Args:
            **kwargs: Keyword arguments passed to the OpenAI API's chat completions method.
        
        Returns:
            openai.ChatCompletionResponse: The response object from the OpenAI API 
            containing the generated completions.
        
        Raises:
            openai.OpenAIError: If the OpenAI API request fails after the specified retries.
        """
        return self.model.chat.completions.create(**kwargs)
