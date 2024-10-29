import torch
import copy
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
from io import BytesIO
from urllib.request import urlopen
import librosa
import os
from .BaseWrapper import BaseWrapper
from ..utils.chat_template import apply_chat_template
from ..utils.model import get_model
from ..utils.utils import is_local


class HFWrapper(BaseWrapper):
    def __init__(self, config, generation_config, template=None):
        self.model, self.tokenizer = get_model(config=config)
        self.model.eval()
        self.config = config
        self.generation_config = generation_config
        self.model_template = template

    def __call__(self, prompts, return_probs=False):
        if self.config.model_name == "Qwen/Qwen2-Audio-7B-Instruct":
            texts = [self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) for prompt in prompts]
            audios = []
            for prompt in prompts:
                audio = []
                for message in prompt:
                    if isinstance(message["content"], list):
                        for ele in message["content"]:
                            if ele["type"] == "audio":
                                audio_file = ele['audio_url']
                                if is_local(ele['audio_url']):
                                    audio_file = "file://"+os.path.abspath(ele['audio_url'])
                               
                                audio.append(
                                    librosa.load(
                                        BytesIO(urlopen(audio_file).read()), 
                                        sr=self.tokenizer.feature_extractor.sampling_rate)[0]
                                )
                audios.append(audio)
            processed_prompts = [self.tokenizer(text=text, audios=audio, return_tensors="pt").to(self.model.device) for text, audio in zip(texts,audios)]
        else:
            processed_prompts = apply_chat_template(prompts, self.model_template)
            processed_prompts = [self.tokenizer(prompt, return_tensors="pt").to(
                self.model.device) for prompt in prompts]
            
        # print(prompts[0])
        # exit(0)
        generations = []
        generations_probs = []
        num_generated_tokens = []
        for idx, prompt in enumerate(processed_prompts):
            inputs = prompt
            try:
                with torch.no_grad():
                    generate_dict = self.model.generate(
                        output_scores=True,
                        return_dict_in_generate=True,
                        **inputs,
                        **self.generation_config,
                    )
            except Exception as e:
                print(prompts[idx])
                raise e
            num_generated_token = len(generate_dict.scores)
            num_generated_tokens.append(num_generated_token)
            generated_tokens = generate_dict.sequences[
                :, -num_generated_token:
            ]

            generation = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            generations.extend(generation)

            if return_probs:
                # Inlcude probabilities of '</s>' token
                generation_probs = self.model.compute_transition_scores(
                    sequences=generated_tokens,
                    scores=generate_dict.scores,
                    normalize_logits=True,
                )
                generations_probs.extend(
                    generation_probs.cpu().numpy().tolist()
                )

        return generations, generations_probs, num_generated_tokens

    def compute_logprob_and_length(self, prompts, completions):
        completions_num_tokens = []
        completions_logprobs = []
        prompts = copy.deepcopy(prompts)
        prompts = apply_chat_template(prompts, self.model_template)
        for prompt, completion in zip(prompts, completions):
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt").to(
                self.model.device
            )  # <s> SPIECE_UNDERLINE [tokens]
            # Actual number of tokens in completion (without `<s>`)
            prompt_num_tokens = prompt_tokens.input_ids.shape[1] - 1

            completion_tokens = self.tokenizer(
                f"{completion}{self.tokenizer.eos_token}", return_tensors="pt"
            ).to(
                self.model.device
            )
            completion_num_tokens = completion_tokens.input_ids.shape[1] - 1
            if completion_tokens.input_ids[0, 1] == 29871:
                completion_num_tokens = completion_num_tokens - 1
            completions_num_tokens.append(completion_num_tokens)

            inputs = torch.concatenate(
                (
                    prompt_tokens.input_ids,
                    completion_tokens.input_ids[:, -completion_num_tokens:],
                ),
                dim=-1,
            )
            outputs = self.model(inputs)
            # [input_tokens] [next_token]

            # Include probabilities of 'SPIECE_UNDERLINE </s>' tokens
            logits = outputs.logits[
                :,
                prompt_num_tokens:prompt_num_tokens + completion_num_tokens,
            ]
            logprobs = logits.log_softmax(dim=-1)
            # >>> batch_size, sequence_length, vocab_size

            logprobs = logprobs.gather(
                dim=-1,
                index=completion_tokens.input_ids[
                    :, -completion_num_tokens:
                ].unsqueeze(-1),
            ).squeeze(-1)
            # >>> batch_size, sequence_length
            completions_logprobs.append(logprobs.cpu().numpy().tolist())
        return completions_logprobs, completions_num_tokens
