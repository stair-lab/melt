import torch
import json
import os
import openai
import backoff
from dotenv import load_dotenv
import google.generativeai as genai
import random

load_dotenv()


class GeminiWrapper:
    def __init__(self, model_name=None, generation_config=None):
        safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        
        self.key = os.getenv("GEMINI_KEY")
        dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])
        genai.configure(api_key=self.key)
        generation_config = dictfilt(generation_config, ("top_k", "temperature"))
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        # self.generation_config = generation_config

    def __call__(self, prompts, return_probs=False):
        generations = []
        generations_probs = [torch.tensor([])] * len(prompts)
        num_generated_tokens = []
        for prompt in prompts:
            processed_prompt = [list(p.values())[1] for p in prompt]
          
            concat_prompt = "\n".join(processed_prompt)
            try:

                response = self.chat_completions_with_backoff(concat_prompt)
                generations.append(response.text)
                # num_generated_tokens.append(
                #     response.usage_metadata.candidates_token_count
                # )
                num_generated_tokens.append(0)

            except Exception as e:
                print(str(e))
                print(prompt)
                generations.append("[ERROR]")
                num_generated_tokens.append(0)

        return generations, generations_probs, num_generated_tokens

    def compute_logprob_and_length(self, prompts, completions):
        completions_num_tokens = [0] * len(prompts)
        completions_logprobs = [torch.tensor([])] * len(prompts)
        # Not Implement
        return completions_logprobs, completions_num_tokens

    @backoff.on_exception(backoff.expo, Exception, max_tries=10)
    def chat_completions_with_backoff(self, prompt):
        return self.model.generate_content(prompt)
