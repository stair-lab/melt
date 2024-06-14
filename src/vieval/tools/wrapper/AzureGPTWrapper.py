import torch
import os
import openai
import backoff
from .BaseWrapper import BaseWrapper


class AzureGPTWrapper(BaseWrapper):
    def __init__(self, engine=None, generation_config=None):
        self.generation_config = generation_config
        self.model = openai.AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_KEY"),
            api_version=os.getenv("AZURE_VERSION"),
        )
        self.engine = engine

    def __call__(self, prompts, return_probs=False):
        generations = []
        generations_probs = [torch.tensor([])] * len(prompts)
        num_generated_tokens = []
        for prompt in prompts:

            response = self.chat_completions_with_backoff(
                model=self.engine,
                messages=prompt,
                temperature=self.generation_config["temperature"],
                max_tokens=self.generation_config["max_new_tokens"],
                top_p=0.95,
                frequency_penalty=self.generation_config["repetition_penalty"],
            )

            generations.append(response.choices[0].message.content)
            num_generated_tokens.append(response.usage.completion_tokens)

        return generations, generations_probs, num_generated_tokens

    def compute_logprob_and_length(self, prompts, completions):
        completions_num_tokens = [0] * len(prompts)
        completions_logprobs = [torch.tensor([])] * len(prompts)
        # TODO: Implement when OpenAI support logprobs of sentence
        return completions_logprobs, completions_num_tokens

    @backoff.on_exception(backoff.expo, openai.OpenAIError, max_tries=10)
    def chat_completions_with_backoff(self, **kwargs):
        return self.model.chat.completions.create(**kwargs)
