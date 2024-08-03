from typing import Dict
import numpy as np
import regex
from .basic_metrics import exact_match, f1_score
from .base import BaseMetric
import random
import Levenshtein
import os
import pandas as pd
import string as string_func

escape_dict = {
    "\a": r"\a",
    "\b": r"\b",
    "\f": r"\f",
    "\n": r"\n",
    "\r": r"\r",
    "\t": r"\t",
    "\v": r"\v",
}


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2


class ReasoningMetric(BaseMetric):
    def equal(self, prediction: str, refenrence: str) -> float:
        if prediction == refenrence:
            return 1
        else:
            return 0

    def _has_numbers(self, word: str):
        return any(char.isdigit() for char in word)

    def _clean_word(self, word: str) -> str:
        word = word.replace("$", "").split("=")[-1]
        word = word.replace("'", "")
        while len(word) > 0 and word[-1] != "}" and (not word[-1].isdigit()):
            word = word[:-1]
        if "{" not in word:
            word = word.replace("}", "")
        word = word.replace("[\\", "")
        return word

    def _get_math_final_result(self, text: str, mode="p") -> str:
        text = text.replace("\f", "\\f")
        text = text.replace("\b", "\\b")
        words = text.split(" ")[::-1]
        # pattern = regex.compile(r'\\boxed\{(?:[^{}]|(?R))*\}')
        # res_list = pattern.findall(text)
        # return res_list[0] if res_list else None
        for i, _ in enumerate(words):
            words[i] = self._clean_word(words[i])
        for word in words:
            if "boxed" in word:
                return word

        for word in words:
            if self._has_numbers(word):
                return word

        return "".join(random.choice(string_func.ascii_uppercase) for _ in range(4))

    def _remove_boxed(self, text: str) -> str:
        if "oxed" in text:
            text = text.replace(r'"\boxed{', "")
            text = text.replace(r"\boxed{", "")
            text = text.replace(r"\\boxed{", "")
            text = text.replace("\\boxed{", "")
            text = text.replace("\boxed{", "")
            text = text.replace(r"\boxed{", "")
            if text and text[-1] == "}":
                text = text[:-1]
            text = self._clean_word(text)

        return text

    def evaluate(self, data: Dict, args) -> (Dict, Dict):
        result = {}
        raw_predictions = data["predictions"]

        predictions = [
            self._get_answer(raw_prediction, args) for raw_prediction in raw_predictions
        ]
        references = data["references"]
        references = [
            # self._get_answer("{" + f"'{args.key_answer}'" + ":" + f"'{reference}'" + "}", args)
            self._get_answer(reference, args)
            for reference in references
        ]
        # data["predictions"] = predictions
        # data["references"] = references

        f1_scores = [f1_score(*batch) for batch in zip(references, predictions)]
        ems = [exact_match(*batch) for batch in zip(references, predictions)]

        # print(predictions[:10])
        # print(references[:10])
        if args.task == "math":
            predictions = [
                self._get_math_final_result(prediction) for prediction in predictions
            ]
            references = [
                self._get_math_final_result(reference, "r") for reference in references
            ]

            references = [self._remove_boxed(reference) for reference in references]

            predictions = [self._remove_boxed(pred) for pred in predictions]
            data["processed_predictions"] = predictions
            data["processed_references"] = references
            # del data["generation_probs"]
            # del data["calibration_probs"]
        # print(predictions[:10])
        # print(references[:10])
        equals = [
            is_equiv(prediction, refenrence)
            for prediction, refenrence in zip(predictions, references)
        ]
        data["equals"] = equals
        if "fewshot" in data:
            del data["fewshot"]

        # if 'math' in args.filepath:
        #     result = {
        #         "f1_score": np.array(f1_scores).mean(),
        #         "exact_match": np.array(ems).mean(),
        #     }
        # else:
        result = {
            "f1_score": np.array(f1_scores).mean(),
            "exact_match": np.array(ems).mean(),
            "equality": np.array(equals).mean(),
        }
        return data, result
