import torch
import json
import os
import openai
import backoff 
from dotenv import load_dotenv

load_dotenv()


class GPTPipeline:
    def __init__(self, model=None, tokenizer=None, generation_config=None):
        self.gpt = openai
        self.gpt.api_type = "azure"
        self.gpt.api_base = "https://ura-gpt4.openai.azure.com/"
        self.gpt.api_version = "2023-07-01-preview"
        self.gpt.api_key = os.environ.get("GPT_KEY")
      
        self.generation_config = generation_config
        

    def __call__(self, prompts, return_probs=False):
        generations = []
        generations_probs = [torch.tensor([])]*len(prompts)
        num_generated_tokens = []
        for prompt in prompts:
            prompt_lst = prompt.split("[SYS]")
            if len(prompt_lst) < 2:
                msgs = [{"role": "user", "content": prompt_lst[0]}]
            else:
                msgs = [{"role":"system", "content": prompt_lst[0]},{"role": "user", "content": prompt_lst[1]}]
            response = self.chat_completions_with_backoff(
                engine="testing",
                messages=msgs,
                temperature=self.generation_config["temperature"],
                max_tokens=self.generation_config["max_new_tokens"],
                top_p=0.95,
                frequency_penalty=self.generation_config["repetition_penalty"],
            )
           
            generations.append(response['choices'][0]['message']['content'])
            num_generated_tokens.append(response['usage']['completion_tokens'])
      
        return generations, generations_probs, num_generated_tokens 

    def compute_logprob_and_length(self, prompts, completions):
        completions_num_tokens = [0]*len(prompts)
        completions_logprobs = [torch.tensor([])]*len(prompts)
        # Not Implement
        return completions_logprobs, completions_num_tokens
   
    @backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=10)
    def chat_completions_with_backoff(self, **kwargs):
        return self.gpt.ChatCompletion.create(**kwargs)

class LLaMaPipeline:
    def __init__(self, model, tokenizer, generation_config):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

    def __call__(self, prompts, return_probs=False):
        generations = []
        generations_probs = []
        num_generated_tokens = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt, return_tensors="pt").to(self.model.device)
            try:
                with torch.no_grad():
                    generate_dict = self.model.generate(
                        inputs=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        output_scores=True,
                        return_dict_in_generate=True,
                        eos_token_id=self.tokenizer.eos_token_id,  
                        pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
                        **self.generation_config,
                    )
            except Exception as e:
                print(e)
                print(prompt)
                raise e
            num_generated_token = len(generate_dict.scores)
            num_generated_tokens.append(num_generated_token)
            generated_tokens = generate_dict.sequences[:, -
                                                       num_generated_token:]

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
                generations_probs.extend(generation_probs.cpu().numpy())

        return generations, generations_probs, num_generated_tokens

    def compute_logprob_and_length(self, prompts, completions):
        completions_num_tokens = []
        completions_logprobs = []

        for prompt, completion in zip(prompts, completions):
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt").to(
                self.model.device
            )  # <s> SPIECE_UNDERLINE [tokens]
            # Actual number of tokens in completion (without `<s>`)
            prompt_num_tokens = prompt_tokens.input_ids.shape[1] - 1

            completion_tokens = self.tokenizer(
                f"{completion} {self.tokenizer.eos_token}",
                return_tensors="pt"
            ).to(self.model.device)  # <s> SPIECE_UNDERLINE [tokens] SPIECE_UNDERLINE </s>
            # Actual number of tokens in completion (without `<s> SPIECE_UNDERLINE`)
            completion_num_tokens = completion_tokens.input_ids.shape[1] - 1
            if completion_tokens.input_ids[0, 1] == 29871:
                completion_num_tokens = completion_num_tokens - 1
            completions_num_tokens.append(completion_num_tokens)

            inputs = torch.concatenate(
                (prompt_tokens.input_ids,
                 completion_tokens.input_ids[:, -completion_num_tokens:]), dim=-1
            )
            outputs = self.model(inputs)
            # [input_tokens] [next_token]

            # Include probabilities of 'SPIECE_UNDERLINE </s>' tokens
            logits = outputs.logits[
                :, prompt_num_tokens: prompt_num_tokens + completion_num_tokens
            ]
            logprobs = logits.log_softmax(dim=-1)
            # >>> batch_size, sequence_length, vocab_size

            logprobs = logprobs.gather(
                dim=-1, index=completion_tokens.input_ids[:, -completion_num_tokens:].unsqueeze(-1)
            ).squeeze(-1)
            # >>> batch_size, sequence_length
            completions_logprobs.append(logprobs.cpu().numpy())


class LLaMaTGIPipeline:
    def __init__(self, api_endpoint, generation_config):
        self.api_endpoint = api_endpoint
        self.generation_config = generation_config
    
    def __call__(self, prompts, return_probs=False):
        generations = []
        generations_probs = []
        num_generated_tokens = []
        for prompt in prompts:
            try:
                generate_dict = self.generate_with_backoff(
                    {
                        "inputs": prompt,
                        "parameters": { "details": True, **self.generation_config}
                    }
                )
            except Exception as e:
                print(e)
                print(prompt)
                raise e
            generation, generation_probs, num_generated_token = self.get_text_logprobs_tgi(generate_dict)
          
            num_generated_tokens.extend(num_generated_token)
            generations.extend(generation)

            if return_probs:
                # Inlcude probabilities of '</s>' token
                generations_probs.extend(generation_probs)

        return generations, generations_probs, num_generated_tokens

    def compute_logprob_and_length(self, prompts, completions):
        completions_num_tokens = []
        completions_logprobs = []

        for prompt, completion in zip(prompts, completions):
            try:
                prompt_tokens = self.generate_with_backoff(
                    {
                        "inputs": prompt,
                        "parameters": { "decoder_input_details":True, "max_new_tokens": 1}
                    }
                )['details']['prefill']
                completion_w_prompt = self.generate_with_backoff(
                    {
                        "inputs": prompt + completion + "</s>",
                        "parameters": { "decoder_input_details":True, "max_new_tokens": 1}
                    }
                )['details']['prefill']
            except Exception as e:
                print(e)
                print(prompt)
                raise e
            logprobs = torch.tensor([list(map(lambda x: x['logprob'], completion_w_prompt[len(prompt_tokens): ]))])
            completions_logprobs.append(logprobs)
            completions_num_tokens.append(len(logprobs[0))

        return completions_logprobs, completions_num_tokens
    @backoff.on_exception(backoff.expo, requests.exeptions.RequestException, max_tries=10)
    def generate_with_backoff(self, inputs):
        generate_obj = requests.post(url+"/generate", json = inputs, verify=False)
        return generate_obj.json()
        
    def get_text_logprobs_tgi(self, res):
        return [res['generated_text']],[torch.tensor(list(map(lambda x: x['logprob'], res['details']['tokens'])))], [res['details']['generated_tokens']]
