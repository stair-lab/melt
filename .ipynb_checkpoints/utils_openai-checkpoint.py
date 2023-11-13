import tiktoken
from typing import List

price_prompt_tokens = {
    'gpt-3.5-turbo': 0.0015 * 0.001,
    'gpt-3.5-turbo-16k': 0.003 * 0.001,
    'gpt-4': 0.03 * 0.001,
    'gpt-4-32k': 0.06 * 0.001,
}

price_completion_tokens = {
    'gpt-3.5-turbo': 0.002 * 0.001,
    'gpt-3.5-turbo-16k': 0.004 * 0.001,
    'gpt-4': 0.06 * 0.001,
    'gpt-4-32k': 0.12 * 0.001,
}

def num_tokens_from_string(text: str, encoding_name: str = 'gpt-3.5-turbo', role="user") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def num_tokens_from_messages(messages: List[dict], encoding_name: str = 'gpt-3.5-turbo') -> int:
    num_tokens = 3
    for message in messages:
        content = message['content']
        role = message['role']
        num_tokens += 4
        num_tokens += num_tokens_from_string(content, encoding_name, role)

    return num_tokens

def usage_token_from_messages(messages: List[dict], encoding_name: str = 'gpt-3.5-turbo', generation_config: dict = {}) -> dict:
    num_prompt_tokens = num_tokens_from_messages(messages, encoding_name)
    num_completion_tokens = generation_config.get('max_new_tokens', 100)
    return {
    "prompt_tokens": num_prompt_tokens,
    "completion_tokens": num_completion_tokens,
    "total_tokens": num_prompt_tokens + num_completion_tokens,
    "cost": num_prompt_tokens * price_prompt_tokens[encoding_name] + num_completion_tokens * price_completion_tokens[encoding_name]
    }

def usage_token_from_prompts(user_prompts: List[str], system_prompt: str = '', encoding_name: str = 'gpt-3.5-turbo', generation_config: dict = {}) -> (int, float):
    total_tokens = 0
    cost = 0.0
    for prompt in user_prompts:
        if system_prompt:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt},
            ]
        else:
            messages = [{'role': 'user', 'content': prompt}]

        usage = usage_token_from_messages(messages, encoding_name, generation_config)
        total_tokens += usage['total_tokens']
        cost += usage['cost']

    return total_tokens, cost

if __name__ == '__main__':

    messages2 =[{"role":"system", "content": "You are an AI bot and you are very smart."}, 
                {"role": "user", "content": "Hi my name is TOan"}]

    print(usage_token_from_messages(messages2))

    print(usage_token_from_prompts(["Make a poem", "List 5 odd numbers"]))

