from dataclasses import dataclass


@dataclass
class GenerationConfig:
    text_generation = {
        "temperature": 0.0,
        "max_new_tokens": 500,
        "repetition_penalty": 1.1,
    }

    question_answering = {
        "temperature": 0.0,
        "max_new_tokens": 100,
        "repetition_penalty": 1.1,
    }

    summarization = {
        "temperature": 0.0,
        "max_new_tokens": 300,
        "repetition_penalty": 1.1,
    }

    translation = {
        "temperature": 0.0,
        "max_new_tokens": 500,
        "repetition_penalty": 1.1,
    }
