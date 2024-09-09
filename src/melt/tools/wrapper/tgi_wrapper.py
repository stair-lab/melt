"""
This module provides a wrapper class for interacting with the TGI API.

It includes the `TGIWrapper` class that facilitates text generation and analysis
using the TGI model. The class supports features such as:
- Generating text from prompts.
- Computing log probabilities and lengths of completions.
- Fetching model information.
- Handling retries with exponential backoff for API requests.

Dependencies:
- `requests`: For making HTTP requests to the TGI API.
- `backoff`: For implementing retry logic.
- `transformers`: For loading the model tokenizer.
- `utils.chat_template`: For applying chat templates to prompts.

Usage:
    Instantiate the `TGIWrapper` with the required configuration and optional template,
    then use its methods to generate text, compute log probabilities, 
    and retrieve model information.
"""
import os
import copy
import backoff
import requests
from transformers import AutoTokenizer
from utils.chat_template import apply_chat_template
from .BaseWrapper import BaseWrapper



class TGIWrapper(BaseWrapper):
    """
    A wrapper class for interacting with the TGI API.

    Attributes:
        generation_config: Configuration parameters for text generation.
        template: Optional template for the model.
        model_info: Information about the model obtained from the API.
        tokenizer: Tokenizer for the model.
    """
    def __init__(self, generation_config, template=""):
        self.api_endpoint = os.getenv("TGI_ENDPOINT")
        self.generation_config = generation_config
        self.model_template = template
        self.model_info = self.get_model_info()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_info["model_id"], trust_remote_code=True
        )

    def __call__(self, prompts, return_probs=False):
        """
        Generate text from prompts.

        Args:
            prompts (list): List of prompts to generate text for.
            return_probs (bool): Whether to return probabilities of generated tokens.

        Returns:
            tuple: Generated texts, probabilities of 
            generated tokens, and number of generated tokens.
        """
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
        """
        Fetches model information from the API endpoint.

        Returns:
            dict: The JSON response from the API, containing model information.
        """
        response = requests.get(
            self.api_endpoint + "/info",
            verify=False,
            timeout=10  # Added timeout argument to prevent indefinite hanging
        )
        return response.json()

    @backoff.on_exception(
        backoff.expo, requests.exceptions.RequestException, max_tries=10
    )
    def generate_with_backoff(self, inputs):
        """
        Sends a request to generate text with retry logic.

        Args:
            inputs (dict): Input parameters for text generation.

        Returns:
            dict: Response from the generation API, containing the generated text and other details.
        """
        response = requests.post(
            self.api_endpoint + "/generate",
            json=inputs,
            verify=False,
            timeout=10  # Added timeout argument to prevent indefinite hanging
        )
        return response.json()

    def get_text_logprobs_tgi(self, res):
        """
        Extracts generated text, log probabilities, 
        and number of generated tokens from the API response.

        Args:
            res (dict): Response dictionary from the API, which includes:
                - "generated_text": The text generated by the model.
                - "details": A dictionary containing:
                    - "tokens": List of token dictionaries, each containing "logprob".
                    - "generated_tokens": Number of tokens generated.

        Returns:
            tuple: A tuple containing:
                - A list with the generated text.
                - A list with a list of log probabilities for each token.
                - A list with the number of generated tokens.
        """
        return (
            [res["generated_text"]],
            [list(map(lambda x: x["logprob"], res["details"]["tokens"]))],
            [res["details"]["generated_tokens"]],
        )
