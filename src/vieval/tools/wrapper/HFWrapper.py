import torch
from .BaseWrapper import BaseWrapper
from ..utils.chat_template import apply_chat_template
from ..utils.model import get_model


class HFWrapper(BaseWrapper):
    def __init__(self, config, generation_config, template=None):
        self.model, self.tokenizer = get_model(config=config)
        self.model.eval()

        self.generation_config = generation_config
        self.model_template = template

    def __call__(self, prompts, return_probs=False):
        generations = []
        generations_probs = []
        num_generated_tokens = []
        prompts = apply_chat_template(prompts, self.model_template)
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            try:
                with torch.no_grad():
                    generate_dict = self.model.generate(
                        inputs=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        output_scores=True,
                        return_dict_in_generate=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=(
                            self.tokenizer.pad_token_id
                            if self.tokenizer.pad_token_id
                            else self.tokenizer.eos_token_id
                        ),
                        **self.generation_config,
                    )
            except Exception as e:
                print(prompt)
                raise e
            num_generated_token = len(generate_dict.scores)
            num_generated_tokens.append(num_generated_token)
            generated_tokens = generate_dict.sequences[:, -num_generated_token:]

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
                generations_probs.extend(generation_probs.cpu().numpy().tolist())

        return generations, generations_probs, num_generated_tokens

    def compute_logprob_and_length(self, prompts, completions):
        completions_num_tokens = []
        completions_logprobs = []
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
            )  # <s> SPIECE_UNDERLINE [tokens] SPIECE_UNDERLINE </s>
            # Actual number of tokens in completion (without `<s> SPIECE_UNDERLINE`)
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
                :, prompt_num_tokens : prompt_num_tokens + completion_num_tokens
            ]
            logprobs = logits.log_softmax(dim=-1)
            # >>> batch_size, sequence_length, vocab_size

            logprobs = logprobs.gather(
                dim=-1,
                index=completion_tokens.input_ids[:, -completion_num_tokens:].unsqueeze(
                    -1
                ),
            ).squeeze(-1)
            # >>> batch_size, sequence_length
            completions_logprobs.append(logprobs.cpu().numpy().tolist())
        return completions_logprobs, completions_num_tokens
