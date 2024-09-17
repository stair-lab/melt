"""
This module provides functions to calculate token usage 
and associated costs for different GPT models.

Functions:
- num_tokens_from_string(text: str, encoding_name: str = "gpt-3.5-turbo", role="user") -> int:
  Returns the number of tokens in a text string based on the specified encoding model.

- num_tokens_from_messages(messages: List[dict], encoding_name: str = "gpt-3.5-turbo") -> int:
  Calculates the total number of tokens for a list of message dictionaries.

- usage_token_from_messages(messages: List[dict], encoding_name: str = 
"gpt-3.5-turbo", generation_config: dict = {}) -> dict:
  Computes the token usage and cost for a list of message dictionaries.

- usage_token_from_prompts(user_prompts: List[str], encoding_name: 
str = "gpt-3.5-turbo", generation_config: dict = {}) -> (dict, float):
  Calculates the token usage and cost for a list of user 
  prompts, accounting for system and user roles.

Dependencies:
- tiktoken: A library for encoding text to tokens for specific GPT models.
- typing: For type annotations used in function signatures.
"""
from typing import List

try:
    import tiktoken
except ImportError:
    print("The 'tiktoken' library is not installed.")
    # Optionally, you could handle the error more gracefully or provide alternative code paths
    tiktoken = None


price_prompt_tokens = {
    "gpt-3.5-turbo": 0.0015 * 0.001,
    "gpt-3.5-turbo-16k": 0.003 * 0.001,
    "gpt-4": 0.03 * 0.001,
    "gpt-4-32k": 0.06 * 0.001,
}

price_completion_tokens = {
    "gpt-3.5-turbo": 0.002 * 0.001,
    "gpt-3.5-turbo-16k": 0.004 * 0.001,
    "gpt-4": 0.06 * 0.001,
    "gpt-4-32k": 0.12 * 0.001,
}


def num_tokens_from_string(
    text: str, encoding_name: str = "gpt-3.5-turbo"
) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def num_tokens_from_messages(
    messages: List[dict], encoding_name: str = "gpt-3.5-turbo"
) -> int:
    """
    Calculates the total number of tokens for a list of message dictionaries.

    Args:
        messages (List[dict]): A list of message dictionaries where each dictionary contains:
            - 'role': A string indicating the role of the message sender (e.g., 'system', 'user').
            - 'content': A string representing the content of the message.
        encoding_name (str): The name of the encoding model to use (default is 'gpt-3.5-turbo').

    Returns:
        int: The total number of tokens for the provided messages.
    """
    num_tokens = 3
    for message in messages:
        content = message["content"]
        num_tokens += 4
        num_tokens += num_tokens_from_string(content, encoding_name)  # Removed the 'role' argument

    return num_tokens


def usage_token_from_messages(
    messages: List[dict],
    encoding_name: str = "gpt-3.5-turbo",
    generation_config: dict = None,  # Change default to None
) -> dict:
    """
    Computes token usage and associated cost for a list of message dictionaries.

    Args:
        messages (List[dict]): A list of message dictionaries, where each dictionary contains:
            - 'role': A string indicating the role of the message sender (e.g., 'system', 'user').
            - 'content': A string representing the content of the message.
        encoding_name (str): The name of the encoding model to use (default is 'gpt-3.5-turbo').
        generation_config (dict, optional): Configuration dictionary for generation settings. 
            It can include 'max_new_tokens' to specify the 
            maximum number of completion tokens (default is 100).

    Returns:
        dict: A dictionary containing:
            - 'prompt_tokens': The number of tokens in the prompt messages.
            - 'completion_tokens': The number of tokens for 
            the completion, based on the 'max_new_tokens' setting.
            - 'total_tokens': The total number of tokens (prompt + completion).
            - 'cost': The cost of using the tokens, calculated based on predefined pricing.
    """
    if generation_config is None:
        generation_config = {}  # Initialize the default mutable object

    num_prompt_tokens = num_tokens_from_messages(messages, encoding_name)
    num_completion_tokens = generation_config.get("max_new_tokens", 100)
    return {
        "prompt_tokens": num_prompt_tokens,
        "completion_tokens": num_completion_tokens,
        "total_tokens": num_prompt_tokens + num_completion_tokens,
        "cost": num_prompt_tokens * price_prompt_tokens[encoding_name]
        + num_completion_tokens * price_completion_tokens[encoding_name],
    }



def usage_token_from_prompts(
    user_prompts: List[str],
    encoding_name: str = "gpt-3.5-turbo",
    generation_config: dict = None,  # Change default to None
) -> (dict, float):
    """
    Computes the total token usage and associated cost for a list of user prompts.

    Args:
        user_prompts (List[str]): A list of user prompts. Each prompt may contain a 
        '[SYS]' separator
            to distinguish between system and user messages.
        encoding_name (str): The name of the encoding model to use (default is 'gpt-3.5-turbo').
        generation_config (dict, optional): Configuration dictionary for generation settings.
            It can include 'max_new_tokens' to specify the maximum number of completion 
            tokens (default is 100).

    Returns:
        tuple: A tuple containing:
            - A dictionary with the following keys:
                - 'total_tokens': The total number of tokens (sum of prompt and completion tokens).
                - 'prompt_tokens': The number of tokens in the input prompts.
                - 'completion_tokens': The number of tokens in the completion 
                (based on 'max_new_tokens').
            - A float representing the total cost of the tokens, calculated based 
            on predefined pricing.
    """
    if generation_config is None:
        generation_config = {}  # Initialize the default mutable object

    total_tokens = 0
    input_tokens = 0
    output_tokens = 0
    cost = 0.0
    for prompt in user_prompts:
        sys_user_prompt = prompt.split("[SYS]")
        if len(sys_user_prompt) > 1:
            messages = [
                {"role": "system", "content": sys_user_prompt[0]},
                {"role": "user", "content": sys_user_prompt[1]},
            ]
        else:
            messages = [{"role": "user", "content": sys_user_prompt[0]}]

        usage = usage_token_from_messages(
            messages, encoding_name, generation_config
        )
        input_tokens += usage["prompt_tokens"]
        output_tokens += usage["completion_tokens"]
        total_tokens += usage["total_tokens"]
        cost += usage["cost"]

    return {
        "total_tokens": total_tokens,
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
    }, cost



if __name__ == "__main__":
    messages2 = [
        {
            "role": "system",
            "content": "You are an AI bot and you are very smart.",
        },
        {"role": "user", "content": "Hi my name is TOan"},
    ]

    print(usage_token_from_messages(messages2))

    print(usage_token_from_prompts(["Make a poem", "List 5 odd numbers"]))
