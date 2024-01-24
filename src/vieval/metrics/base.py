from .post_process import get_answer_auto_from_text


class BaseMetric:
    def __init__(self):
        return

    def _get_answer(self, text: str, args) -> str:
        return get_answer_auto_from_text(
            text=text,
            key_answer=args.key_answer,
            class_names=args.class_names,
            args=args,
        )
