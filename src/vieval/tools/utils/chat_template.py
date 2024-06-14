from typing import Union, List, Optional, Dict
from packaging import version


def _compile_jinja_template(chat_template):
    try:
        import jinja2
        from jinja2.exceptions import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        raise ImportError("apply_chat_template requires jinja2 to be installed.")

    if version.parse(jinja2.__version__) < version.parse("3.0.0"):
        raise ImportError(
            "apply_chat_template requires jinja2>=3.0.0 to be installed. Your version is "
            f"{jinja2.__version__}."
        )

    def raise_exception(message):
        raise TemplateError(message)

    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    jinja_env.globals["raise_exception"] = raise_exception
    return jinja_env.from_string(chat_template)


def apply_chat_template(
    conversation: Union[
        List[Dict[str, str]], List[List[Dict[str, str]]], "Conversation"
    ],
    chat_template: Optional[str] = None,
    add_generation_prompt: bool = True,
) -> Union[str]:

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
                    conversation[0]["content"] + "\n\n" + conversation[1]["content"]
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


if __name__ == "__main__":
    chat = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"},
    ]
    chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '<s>' + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + '</s>' }}{% endif %}{% endfor %}"
    a = apply_chat_template(chat, chat_template)
    print(a)
