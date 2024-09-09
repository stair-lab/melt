"""
This module provides a wrapper class for interacting with the Gemini
Generative Model API, including methods for generating content and
handling prompts.
"""

import os
try:
    import backoff
except ModuleNotFoundError as e:
    print(f"Module 'backoff' not found: {e}")
except ImportError as e:
    print(f"Failed to import the 'backoff' module: {e}")



try:
    import google.generativeai as genai
except ModuleNotFoundError as e:
    print(f"Module 'google.generativeai' not found: {e}")
except ImportError as e:
    print(f"Failed to import 'google.generativeai': {e}")





class GeminiWrapper:
    """
    Wrapper class for the Gemini Generative Model API.

    This class provides methods to interact with the Gemini Generative Model,
    including generating content from prompts and handling safety settings.
    """
    def __init__(self, model_name=None, generation_config=None):
        """Initialize the GeminiWrapper with the specified model and configuration.

        Args:
            model_name (str, optional): The name of the model to use.
            generation_config (dict, optional): Configuration parameters for content generation.
        """
        safety_settings = [
            {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        self.key = os.getenv("GEMINI_KEY")

        def dictfilt(x, y):
            return {i: x[i] for i in x if i in set(y)}
        genai.configure(api_key=self.key)
        generation_config = dictfilt(
            generation_config or {},
            ("top_k", "temperature")
        )
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

    def __call__(self, prompts, return_probs=False):
        """Generate responses for the given prompts.

        Args:
            prompts (list of list of dict): The prompts to be processed.
            return_probs (bool, optional): Whether to return probabilities. Defaults to False.

        Returns:
            tuple: A tuple containing the generations, generation probabilities, 
            and number of generated tokens.
        """
        generations = []
        generations_probs = [[] for _ in prompts]
        num_generated_tokens = []
        for prompt in prompts:
            processed_prompt = [list(p.values())[1] for p in prompt]
            concat_prompt = "\n".join(processed_prompt)
            try:
                response = self.chat_completions_with_backoff(concat_prompt)
                generations.append(response.text)
                num_generated_tokens.append(0)
            except (backoff.BackoffError, genai.GenerativeAIError) as e:
                print(str(e))
                print(prompt)
                generations.append("[ERROR]")
                num_generated_tokens.append(0)

        return generations, generations_probs, num_generated_tokens

    def compute_logprob_and_length(self, prompts):
        """Compute the log probabilities and lengths for given prompts and completions.

        Args:
            prompts (list of list of dict): The prompts to evaluate.
            completions (list of str): The completions to evaluate.

        Returns:
            tuple: A tuple containing the log probabilities and lengths.
        """
        completions_num_tokens = [0] * len(prompts)
        completions_logprobs = [[] for _ in prompts]
        # Not Implemented
        return completions_logprobs, completions_num_tokens

    @backoff.on_exception(backoff.expo, Exception, max_tries=10)
    def chat_completions_with_backoff(self, prompt):
        """Generate content with retry logic on failure.

        Args:
            prompt (str): The prompt to generate content for.

        Returns:
            google.generativeai.GenerativeResponse: The response from the model.
        """
        return self.model.generate_content(prompt)
