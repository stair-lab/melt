"""
This module provides utilities for applying a chat template using Jinja2.
The chat template is rendered based on conversation data.
"""

from typing import Union, List, Optional, Dict
from packaging import version
import jinja2
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment


def _compile_jinja_template(chat_template):
    """
    Compiles the given Jinja2 chat template.

    Args:
        chat_template (str): The chat template in string format.

    Returns:
        Template: A compiled Jinja2 template object.

    Raises:
        ImportError: If Jinja2 is not installed or the version is less than 3.0.0.
    """
    if version.parse(jinja2.__version__) < version.parse("3.0.0"):
        raise ImportError(
            "apply_chat_template requires jinja2>=3.0.0 to be installed. "
            f"Your version is {jinja2.__version__}."
        )

    def raise_exception(message):
        raise TemplateError(message)

    jinja_env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    )
    jinja_env.globals["raise_exception"] = raise_exception
    return jinja_env.from_string(chat_template)


def apply_chat_template(
    conversation: Union[
        List[Dict[str, str]], List[List[Dict[str, str]]]
    ],
    chat_template: Optional[str] = None,
    add_generation_prompt: bool = True,
) -> Union[str]:
    """
    Applies a chat template to a conversation or a list of conversations.

    Args:
        conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]): The conversation data.
        chat_template (Optional[str]): The Jinja2 chat template to use.
        add_generation_prompt (bool): Whether to add a generation prompt to the output.

    Returns:
        Union[str]: The rendered chat conversation(s) as a string.
    """
    # Compilation function uses a cache to avoid recompiling the same template
    compiled_template = _compile_jinja_template(chat_template["template"])

    if isinstance(conversation, (list, tuple)) and (
        isinstance(conversation[0], (list, tuple))
        or hasattr(conversation[0], "messages")
    ):
        conversations = conversation
        if not chat_template["system_prompt"]:
            for chat in conversations:
                if chat[0]["content"] != "":
                    chat[1]["content"] = (
                        chat[0]["content"] + "\n\n" + chat[1]["content"]
                    )
                del chat[0]
        is_batched = True
    else:
        if not chat_template["system_prompt"]:
            if conversation[0]["content"] != "":
                conversation[1]["content"] = (
                    conversation[0]["content"]
                    + "\n\n"
                    + conversation[1]["content"]
                )
            del conversation[0]
        conversations = [conversation]
        is_batched = False

    rendered = []

    for chat in conversations:
        if hasattr(chat, "messages"):
            # Indicates it's a Conversation object
            chat = chat.messages
        rendered_chat = compiled_template.render(
            messages=chat, add_generation_prompt=add_generation_prompt
        )
        rendered.append(rendered_chat)

    if not is_batched:
        rendered = rendered[0]

    return rendered
