import backoff
import requests
from transformers import AutoTokenizer
import os
import copy
from .BaseWrapper import BaseWrapper
from ..utils.chat_template import apply_chat_template


class TGIWrapper(BaseWrapper):
    def __init__(self, generation_config, template=""):
        self.api_endpoint = os.getenv("TGI_ENDPOINT")
        self.generation_config = generation_config
        self.model_template = template
        self.model_info = self.get_model_info()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_info["model_id"], trust_remote_code=True
        )

    def __call__(self, prompts, return_probs=False):
        generations = []
        generations_probs = []
        num_generated_tokens = []
        prompts = copy.deepcopy(prompts)
        prompts = apply_chat_template(prompts, self.model_template)
        for prompt in prompts:
            try:
                generate_dict = self.generate_with_backoff(
                    {
                        "inputs": prompt,
                        "parameters": {
                            "truncate": self.model_info["max_input_tokens"],
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
        prompts = copy.deepcopy(prompts)
        prompts = apply_chat_template(prompts, self.model_template)
        # tokenized_prompts = self.tokenizer(prompts)["input_ids"]
        # len_tokenized_prompts = [len(p) for p in tokenized_prompts]
        for prompt, completion in zip(prompts, completions):
            try:
                for prompt, completion in zip(prompts, completions):
                    prompt_tokens = self.generate_with_backoff(
                        {
                            "inputs": prompt,
                            "parameters": {
                                "truncate": self.model_info[
                                    "max_input_tokens"
                                ],
                                "decoder_input_details": True,
                                "max_new_tokens": 1,
                            },
                        }
                    )["details"]["prefill"]
                    completion_w_prompt = self.generate_with_backoff(
                        {
                            "inputs": prompt
                            + str(completion)
                            + self.tokenizer.eos_token,
                            "parameters": {
                                "truncate": self.model_info[
                                    "max_input_tokens"
                                ],
                                "decoder_input_details": True,
                                "max_new_tokens": 1,
                            },
                        }
                    )["details"]["prefill"]
            except Exception as e:
                print(e)
                print(prompt)
                raise e
            logprobs = [
                list(
                    map(
                        lambda x: x["logprob"],
                        completion_w_prompt[len(prompt_tokens):],
                    )
                )
            ]
            completions_logprobs.append(logprobs)
            completions_num_tokens.append(len(logprobs[0]))

        return completions_logprobs, completions_num_tokens

    def get_model_info(self):
        info = requests.get(self.api_endpoint + "/info", verify=False)
        return info.json()

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
            [list(map(lambda x: x["logprob"], res["details"]["tokens"]))],
            [res["details"]["generated_tokens"]],
        )
