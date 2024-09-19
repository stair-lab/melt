"name_detector"
import os
import re
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)
from underthesea import sent_tokenize
import torch
import spacy

# load core english library
nlp = spacy.load("en_core_web_sm")

class NameDetector:
    """Detect names within texts, categorize them, and potentially
    process multiple texts in batches."""

    token_pattern = ""  # Renamed from TOKEN_PATTERN to token_pattern

    def __init__(self, args):
        with open(
            os.path.join(
                args.config_dir, args.lang, "words", "token_pattern.txt"
            ),
            "r",
            encoding="utf-8"
        ) as f:
            self.token_pattern = f.read().strip()  # Updated attribute name here as well
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
        """Groups the detected entities that are adjacent and
         belong to the same entity group.

         Args:
            text (str): The original text from which entities are extracted.

            entities (list): A list of entity dictionaries
            detected in the text.

         Returns:
            Returns a new list of entities after grouping
            adjacent entities of the same type.
        """
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
                new_entity["word"] = text[
                    new_entity["start"]:new_entity["end"]
                ]
                new_entity["score"] = max(
                    new_entity["score"], entities[i]["score"]
                )
            else:
                new_entities.append(new_entity)
                new_entity = entities[i]

        new_entities.append(new_entity)
        return new_entities

    def _get_person_tokens(self, all_tokens):
        """Filters and retrieves tokens classified as persons
        from the detected entities."""
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
        """Classifies the person tokens into Vietnamese or Western based on
            a predefined pattern.

            Args:
            per_tokens (list): A list of person name tokens to be classified.

            Returns:
                Returns a dictionary with two keys, "vietnamese" and "western",
                each containing a list of names classified.
        """
        results = {
            "your_race": set(),
            "western": set(),
        }
        for token in per_tokens:
            if re.search(self.token_pattern, token) is None:  # Updated usage here
                results["western"].add(token)
            else:
                results["your_race"].add(token)

        results["western"] = list(results["western"])
        results["your_race"] = list(results["your_race"])
        return results

    def detect(self, text):
        """Detects and classifies names in a single text string.

            Args:
            text (str): The input text to process.

            Returns:
            Returns a dictionary with classified names.
        """
        sentences = sent_tokenize(text)
        print(len(sentences))
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
        """Detects and classifies names in a batch of text strings.

         Args:
            texts (list): A list of text strings to process in batch.

         Returns:
            Returns a dictionary with classified names for the batch.
        """
        sentences = []

        for text in texts:
            doc = nlp(text)
            sentences.extend([sent.text for sent in doc.sents])

        sentences = [
            " ".join(sentence.split(" ")[: self.max_words_sentence])
            for sentence in sentences
        ]
        entities_lst = self.token_classifier(sentences, batch_size=128)

        all_entities = []
        for sentence, entities in zip(sentences, entities_lst):
            all_entities += self.group_entity(sentence, entities)

        per_tokens = self._get_person_tokens(all_entities)
        names = self._classify_race(per_tokens)
        return names
