"""
This module contains the VLLMWrapper class for interacting with the VLLM model.

The VLLMWrapper class provides methods to generate text, compute log probabilities,
and handle prompts using the VLLM API. It includes functionality for handling model
configurations, applying templates, and processing results.

Classes:
    VLLMWrapper: A wrapper class for the VLLM model, providing text generation and
    log probability computation.

Usage:
    - Initialize with model configuration and generation settings.
    - Call the instance with prompts to generate text.
    - Compute log probabilities and token counts for given completions.

Example:
    config = Config(model_name="my-model", cpu_offload_gb=4, dtype="float32")
    generation_config = {"max_new_tokens": 50, "repetition_penalty": 1.2}
    wrapper = VLLMWrapper(config, generation_config)
    prompts = ["Hello, world!"]
    generations, probs, num_tokens = wrapper(prompts, return_probs=True)
"""

import copy
from typing import Dict

try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError as e:
    print(f"Module 'vllm' not found: {e}")
except ImportError as e:
    print(f"Failed to import 'LLM' or 'SamplingParams' from 'vllm': {e}")



from .BaseWrapper import BaseWrapper

try:
    from ..utils.chat_template import apply_chat_template
except ModuleNotFoundError as e:
    print(f"Module 'utils.chat_template' not found: {e}")
except ImportError as e:
    print(f"Failed to import 'apply_chat_template' from 'utils.chat_template': {e}")



class VLLMWrapper(BaseWrapper):
    """
    A wrapper class for interacting with the VLLM model API.

    This class provides methods to generate text and compute log probabilities
    using the VLLM model. It handles model configuration, applies text templates,
    and processes API responses to return generated text and related information.

    Attributes:
        model (LLM): An instance of the LLM model from the vllm package.
        generation_config (SamplingParams): Configuration parameters for text generation.
        model_template (Dict, optional): Template for processing prompts.

    Methods:
        __call__(prompts, return_probs=False):
            Generates text from prompts and optionally returns log probabilities.
        
        compute_logprob_and_length(prompts, completions):
            Computes the log probabilities and lengths of completions relative to prompts.
    """

    def __init__(self, config, generation_config, template: Dict = None):
        generation_config["max_tokens"] = generation_config.pop(
            "max_new_tokens"
        )
        generation_config["frequency_penalty"] = generation_config.pop(
            "repetition_penalty"
        )
        self.model = LLM(
            model=config.model_name,
            cpu_offload_gb=config.cpu_offload_gb,
            dtype=config.dtype,
        )
        self.generation_config = SamplingParams(
            **generation_config, logprobs=1, prompt_logprobs=0
        )
        self.model_template = template

    def __call__(self, prompts, return_probs=False):
        generations = []
        generations_probs = []
        num_generated_tokens = []
        prompts = copy.deepcopy(prompts)
        prompts = apply_chat_template(prompts, self.model_template)
        try:
            outputs = self.model.generate(prompts, self.generation_config)
            for output in outputs:
                generations.append(output.outputs[0].text)
                generations_probs.append(
                    [
                        list(logprob.values())[0].logprob
                        for logprob in output.outputs[0].logprobs
                    ]
                )
                num_generated_tokens.append(len(output.outputs[0].logprobs))
        except Exception as e:
            print(prompts)
            raise e
        return generations, generations_probs, num_generated_tokens

    def compute_logprob_and_length(self, prompts, completions):
        tokenizer = self.model.get_tokenizer()
        completions_num_tokens = []
        completions_logprobs = []
        prompts = copy.deepcopy(prompts)
        prompts = apply_chat_template(prompts, self.model_template)
        tokenized_prompts = tokenizer(prompts)["input_ids"]
        len_tokenized_prompts = [len(p) for p in tokenized_prompts]
        completed_prompts = [
            prompt + str(completion) + tokenizer.eos_token
            for prompt, completion in zip(prompts, completions)
        ]
        outputs = self.model.generate(
            completed_prompts,
            SamplingParams(
                max_tokens=1,
                prompt_logprobs=0,
                ignore_eos=False,
                skip_special_tokens=False,
            ),
        )
        for output, len_tokenized_prompt in zip(
            outputs, len_tokenized_prompts
        ):
            completions_num_tokens.append(
                len(output.prompt_logprobs) - len_tokenized_prompt
            )
            completions_logprobs.append(
                [
                    [
                        list(logprob.values())[0].logprob
                        for logprob in output.prompt_logprobs[
                            len_tokenized_prompt:
                        ]
                    ]
                ]
            )
        return completions_logprobs, completions_num_tokens
