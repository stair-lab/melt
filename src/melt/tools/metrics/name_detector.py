"""
This module provides functionality for detecting names in text using natural
language processing techniques.
"""

import os
import re

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from underthesea import sent_tokenize
import torch
import spacy

# Load the core English NLP library
nlp = spacy.load("en_core_web_sm")


class NameDetector:
    """Detect names within texts, categorize them, and potentially
    process multiple texts in batches."""

    def __init__(self, args):
        # Use an instance variable instead of a global variable
        with open(
            os.path.join(args.config_dir, args.lang, "words", "token_pattern.txt"),
            "r",
            encoding="utf-8",  # Specify the encoding explicitly
        ) as f:
            self.token_pattern = f.read().strip()  # Store in instance variable

        tokenizer = AutoTokenizer.from_pretrained(
            args.metric_config["NERModel"],
        )
        model = AutoModelForTokenClassification.from_pretrained(
            args.metric_config["NERModel"]
        )
        self.token_classifier = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.max_words_sentence = 200
        self.threshold_score = 0.97
        self.threshold_len = 2

    def group_entity(self, text, entities):
        """Groups adjacent detected entities belonging to the same entity group."""
        if len(entities) == 0:
            return []
        new_entity = entities[0]
        new_entities = []
        for i in range(1, len(entities)):
            if (
                new_entity["end"] == entities[i]["start"]
                and new_entity["entity_group"] == entities[i]["entity_group"]
            ):
                new_entity["end"] = entities[i]["end"]
                new_entity["word"] = text[new_entity["start"] : new_entity["end"]]
                new_entity["score"] = max(new_entity["score"], entities[i]["score"])
            else:
                new_entities.append(new_entity)
                new_entity = entities[i]

        new_entities.append(new_entity)
        return new_entities

    def _get_person_tokens(self, all_tokens):
        """Filters and retrieves person tokens from detected entities."""
        per_tokens = []
        temp = [
            entity
            for entity in all_tokens
            if entity["entity_group"] == "PERSON"
            and len(entity["word"]) > self.threshold_len
            and entity["score"] > self.threshold_score
        ]
        per_tokens.extend([entity["word"] for entity in temp])
        return per_tokens

    def _classify_race(self, per_tokens):
        """Classifies names into Vietnamese or Western categories."""
        results = {
            "your_race": set(),
            "western": set(),
        }
        for token in per_tokens:
            if re.search(self.token_pattern, token) is None:  # Use instance variable
                results["western"].add(token)
            else:
                results["your_race"].add(token)

        results["western"] = list(results["western"])
        results["your_race"] = list(results["your_race"])
        return results

    def detect(self, text):
        """Detects and classifies names in a single text."""
        sentences = sent_tokenize(text)
        sentences = [
            " ".join(sentence.split(" ")[: self.max_words_sentence])
            for sentence in sentences
        ]

        entities_lst = self.token_classifier(sentences)
        all_entities = []
        for sentence, entities in zip(sentences, entities_lst):
            all_entities += self.group_entity(sentence, entities)

        per_tokens = self._get_person_tokens(all_entities)
        names = self._classify_race(per_tokens)
        return names

    def detect_batch(self, texts):
        """Detects and classifies names in a batch of text strings."""
        all_entities = []
        sentences = []

        for text in texts:
            doc = nlp(text)
            sentences = [sent.text for sent in doc.sents]

        sentences = [
            " ".join(sentence.split(" ")[: self.max_words_sentence])
            for sentence in sentences
        ]
        entities_lst = self.token_classifier(sentences, batch_size=128)

        for sentence, entities in zip(sentences, entities_lst):
            all_entities += self.group_entity(sentence, entities)

        per_tokens = self._get_person_tokens(all_entities)
        names = self._classify_race(per_tokens)
        return names
