import openai
import backoff
from .BaseWrapper import BaseWrapper


class OpenAIWrapper(BaseWrapper):
    def __init__(self, engine=None, generation_config=None):
        generation_config["max_tokens"] = generation_config.pop("max_new_tokens")
        generation_config["frequency_penalty"] = generation_config.pop(
            "repetition_penalty"
        )
        self.generation_config = generation_config
        self.model = openai.OpenAI()
        self.engine = engine

    def __call__(self, prompts, return_probs=False):
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
        completions_num_tokens = [0] * len(prompts)
        completions_logprobs = [[]] * len(prompts)
        # TODO: Implement when OpenAI support logprobs of sentence
        return completions_logprobs, completions_num_tokens

    @backoff.on_exception(backoff.expo, openai.OpenAIError, max_tries=10)
    def chat_completions_with_backoff(self, **kwargs):
        return self.model.chat.completions.create(**kwargs)
