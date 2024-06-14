from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from underthesea import sent_tokenize
import re
import spacy

# load core english library
nlp = spacy.load("en_core_web_sm")
vi_pattern = "[ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễếệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]"


class NameDetector:
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "NlpHUST/ner-vietnamese-electra-base", add_special_tokens=True
        )
        model = AutoModelForTokenClassification.from_pretrained(
            "NlpHUST/ner-vietnamese-electra-base"
        ).to("cuda:0")
        self.token_classifier = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device="cuda:0",
        )
        self.max_words_sentence = 200
        self.threshold_score = 0.97
        self.threshold_len = 2

    def group_entity(self, text, entities):
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
        per_tokens = []
        temp = [
            entity
            for entity in all_tokens
            if entity["entity_group"] == "PERSON"
            and len(entity["word"]) > self.threshold_len
            and entity["score"] > self.threshold_score
        ]
        # print(temp)
        per_tokens.extend([entity["word"] for entity in temp])
        return per_tokens

    def _classify_race(self, per_tokens):
        results = {
            "vietnamese": set(),
            "western": set(),
        }
        for token in per_tokens:
            if re.search(vi_pattern, token) is None:
                results["western"].add(token)
            else:
                results["vietnamese"].add(token)

        results["western"] = list(results["western"])
        results["vietnamese"] = list(results["vietnamese"])
        return results

    def detect(self, text):
        all_entities = []
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
