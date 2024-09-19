"""
This module contains the VLLMWrapper class for interacting with the VLLM model.
"""

import copy
from typing import Dict

try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError as e:
    print(f"Module 'vllm' not found: {e}")
except ImportError as e:
    print(f"Failed to import 'LLM' or 'SamplingParams' from 'vllm': {e}")



from melt.tools.wrapper.base_wrapper import BaseWrapper

try:
    from melt.tools.utils.chat_template import apply_chat_template
except ModuleNotFoundError as e:
    print(f"Module 'utils.chat_template' not found: {e}")
except ImportError as e:
    print(f"Failed to import 'apply_chat_template' from 'utils.chat_template': {e}")



class VLLMWrapper(BaseWrapper):
    """
    A wrapper class for interacting with the VLLM model API.
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
