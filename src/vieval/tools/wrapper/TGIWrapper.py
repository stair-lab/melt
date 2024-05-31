import torch
import backoff
import requests
from .BaseWrapper import BaseWrapper


class TGIPipeline(BaseWrapper):
    def __init__(self, api_endpoint, generation_config, template="llama-2"):
        self.api_endpoint = api_endpoint
        self.generation_config = generation_config
        self.model_template = ChatTemplateStyle[template]

    def __call__(self, prompts, return_probs=False):
        generations = []
        generations_probs = []
        num_generated_tokens = []
        prompts = apply_chat_template(prompts, self.model_template)
        for prompt in prompts:
            try:
                generate_dict = self.generate_with_backoff(
                    {
                        "inputs": prompt,
                        "parameters": {
                            "truncate": 1500,
                            "details": True,
                            **self.generation_config,
                        },
                    }
                )
            except Exception as e:
                print(e)
                print(prompt)
                raise e
            (
                generation,
                generation_probs,
                num_generated_token,
            ) = self.get_text_logprobs_tgi(generate_dict)

            num_generated_tokens.extend(num_generated_token)
            generations.extend(generation)

            if return_probs:
                # Inlcude probabilities of '</s>' token
                generations_probs.extend(generation_probs)

        return generations, generations_probs, num_generated_tokens

    def compute_logprob_and_length(self, prompts, completions):
        completions_num_tokens = []
        completions_logprobs = []
        prompts = apply_chat_template(prompts, self.model_template)
        for prompt, completion in zip(prompts, completions):
            try:
                prompt_tokens = self.generate_with_backoff(
                    {
                        "inputs": prompt,
                        "parameters": {
                            "truncate": 1500,
                            "decoder_input_details": True,
                            "max_new_tokens": 1,
                        },
                    }
                )["details"]["prefill"]
                completion_w_prompt = self.generate_with_backoff(
                    {
                        "inputs": prompt + completion + "</s>",
                        "parameters": {
                            "truncate": 1500,
                            "decoder_input_details": True,
                            "max_new_tokens": 1,
                        },
                    }
                )["details"]["prefill"]
            except Exception as e:
                print(e)
                print(prompt)
                raise e
            logprobs = torch.tensor(
                [
                    list(
                        map(
                            lambda x: x["logprob"],
                            completion_w_prompt[len(prompt_tokens) :],
                        )
                    )
                ]
            )
            completions_logprobs.append(logprobs)
            completions_num_tokens.append(len(logprobs[0]))

        return completions_logprobs, completions_num_tokens

    @backoff.on_exception(
        backoff.expo, requests.exceptions.RequestException, max_tries=10
    )
    def generate_with_backoff(self, inputs):
        generate_obj = requests.post(
            self.api_endpoint + "/generate", json=inputs, verify=False
        )
        return generate_obj.json()

    def get_text_logprobs_tgi(self, res):
        return (
            [res["generated_text"]],
            [torch.tensor(list(map(lambda x: x["logprob"], res["details"]["tokens"])))],
            [res["details"]["generated_tokens"]],
        )
