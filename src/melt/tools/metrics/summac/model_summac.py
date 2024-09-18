"""
mypy: check_untyped_defs = False
###############################################
Source: https://github.com/tingofurro/summac
###############################################
"""
import json
import os
import importlib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
import numpy as np
import torch


from melt.tools.metrics.summac import utils_misc

model_map = {}


def card_to_name(card):
    """
    Converts a model card identifier to its corresponding name.
    Args:
        card (str): The model card identifier.
    Returns:
        str: The corresponding model name if found, otherwise returns the card itself.
    """
    card2name = {v["model_card"]: k for k, v in model_map.items()}
    if card in card2name:
        return card2name[card]
    return card


def name_to_card(name):
    """
    Converts a model name to its corresponding model card identifier.

    Args:
        name (str): The model name.

    Returns:
        str: The corresponding model card identifier if found, otherwise returns the name itself.
    """
    if name in model_map:
        return model_map[name]["model_card"]
    return name


def get_neutral_idx(ent_idx, con_idx):
    """
    Returns the index that is neither the 'entailment' index nor the 'contradiction' index.
    Args:
        ent_idx (int): The index representing 'entailment'.
        con_idx (int): The index representing 'contradiction'.
    Returns:
        int: The index that represents 'neutral', which is neither 'entailment' nor 'contradiction'.
    """
    return list(set([0, 1, 2]) - set([ent_idx, con_idx]))[0]


class SummaCImager:
    """
    A class to handle the imager models for text classification.
    """
    def __init__(
        self,
        model_name="mnli",
        granularity="paragraph",
        config=None,
    ):
        if config is None:
            config = {}
        self.grans = granularity.split("-")

        assert (
            all(
                gran
                in ["paragraph", "sentence", "document", "2sents", "mixed"]
                for gran in self.grans
            )
            and len(self.grans) <= 2
        ), f"Unrecognized `granularity` {granularity}"
        assert (
            model_name in model_map.keys()
        ), f"Unrecognized model name: `{model_name}`"

        self.model_name = model_name
        if model_name != "decomp":
            self.model_card = name_to_card(model_name)
            self.entailment_idx = model_map[model_name]["entailment_idx"]
            self.contradiction_idx = model_map[model_name]["contradiction_idx"]
            self.neutral_idx = get_neutral_idx(
                self.entailment_idx, self.contradiction_idx
            )

        self.granularity = granularity
        self.use_cache = config.get('use_cache', True)
        self.cache_folder = config.get('cache_folder', "/export/share/plaban/summac_cache/")
        self.max_doc_sents = config.get('max_doc_sents', 100)
        self.max_input_length = config.get('max_input_length', 500)
        self.device = config.get('device', "cuda")
        self.cache = {}
        self.model = None  # Lazy loader

        self.tokenizer = None
        self.model = None
    def load_nli(self):
        """
        Loads the NLI model based on the specified model name.
        """
        if self.model_name == "decomp":
            predictor_module = importlib.import_module('allennlp.predictors.predictor')
            predictor_class = getattr(predictor_module, 'Predictor')
            self.model = predictor_class.from_path(
                "https://storage.googleapis.com/allennlp-public-models"
                "/decomposable-attention-elmo-2020.04.09.tar.gz",
                cuda_device=0,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_card
            ).eval()
            self.model.to(self.device).half()

    def split_sentences(self, text):
        """
            Splits the given text into sentences and filters out very short sentences.

            Args:
                text (str): The text to be split into sentences.

            Returns:
                list of str: A list of sentences where each sentence has more than 10 characters.
        """
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent) > 10]
        return sentences
    def split_2sents(self, text):
        """
        Splits the given text into sentences and creates pairs of consecutive sentences.

        Args:
            text (str): The text to be split into sentences.

        Returns:
            list of str: A list of strings where each string is a 
            concatenation of two consecutive sentences
                        from the input text. Each sentence in the list has more than 10 characters.
        """
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent) > 10]
        two_sents = [
            " ".join(sentences[i:(i + 2)]) for i in range(len(sentences))
        ]
        return two_sents
    def split_paragraphs(self, text):
        """
        Splits the input text into paragraphs.
        """
        if text.count("\n\n") > 0:
            paragraphs = [p.strip() for p in text.split("\n\n")]
        else:
            paragraphs = [p.strip() for p in text.split("\n")]
        return [p for p in paragraphs if len(p) > 10]
    def split_text(self, text, granularity="sentence"):
        """
        Splits the given text based on the specified granularity level.
        Args:
            text (str): The text to be split.
            granularity (str): The level of granularity for splitting the text. Options include:
                - "document": Return the entire text as a single element in a list.
                - "paragraph": Split the text into paragraphs.
                - "sentence": Split the text into sentences.
                - "2sents": Split the text into pairs of consecutive sentences.
                - "mixed": Combine sentences and paragraphs from the text.
        Returns:
            list of str: A list of text segments based on the specified granularity.
        Raises:
            ValueError: If the `granularity` argument is not one of the specified options.
        """
        if granularity == "document":
            return [text]
        if granularity == "paragraph":
            return self.split_paragraphs(text)
        if granularity == "sentence":
            return self.split_sentences(text)
        if granularity == "2sents":
            return self.split_2sents(text)
        if granularity == "mixed":
            return self.split_sentences(text) + self.split_paragraphs(text)
        raise ValueError(f"Invalid granularity: {granularity}")

    def build_image(self, original, generated):
        """
        Builds an image representation from the original and generated texts.
        """
        cache_key = (original, generated)
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key][:, :self.max_doc_sents, :]

        gran_doc, gran_sum = self.grans[0], self.grans[-1]

        original_chunks = self.split_text(original, granularity=gran_doc)[:self.max_doc_sents]
        generated_chunks = self.split_text(generated, granularity=gran_sum)

        num_ori = len(original_chunks)
        num_gen = len(generated_chunks)

        if num_ori == 0 or num_gen == 0:
            return np.zeros((3, 1, 1))
        image = np.zeros((3, num_ori, num_gen))

        if self.model is None:
            self.load_nli()

        dataset = [
            {
                "premise": original_chunks[i],
                "hypothesis": generated_chunks[j],
                "doc_i": i,
                "gen_i": j,
            }
            for i in range(num_ori)
            for j in range(num_gen)
        ]

        for batch in utils_misc.batcher(dataset, batch_size=512):
            self._process_batch(batch, image)

        if self.use_cache:
            self.cache[cache_key] = image
        return image

    def _process_batch(self, batch, image):
        if self.model_name == "decomp":
            batch_json = [
                {"premise": d["premise"], "hypothesis": d["hypothesis"]}
                for d in batch
            ]
            model_outs = self.model.predict_batch_json(batch_json)
            for out, b in zip(model_outs, batch):
                probs = out["label_probs"]
                image[0, b["doc_i"], b["gen_i"]] = probs[0]
                image[1, b["doc_i"], b["gen_i"]] = probs[1]
                image[2, b["doc_i"], b["gen_i"]] = probs[2]
        else:
            batch_prems = [b["premise"] for b in batch]
            batch_hypos = [b["hypothesis"] for b in batch]
            batch_tokens = self.tokenizer.batch_encode_plus(
                list(zip(batch_prems, batch_hypos)),
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
                return_tensors="pt",
                truncation_strategy="only_first",
            )
            batch_tokens = {k: v.to(self.device) for k, v in batch_tokens.items()}
            with torch.no_grad():
                model_outputs = self.model(**batch_tokens)

            batch_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)
            for b, probs in zip(batch, batch_probs):
                image[0, b["doc_i"], b["gen_i"]] = probs[self.entailment_idx].item()
                image[1, b["doc_i"], b["gen_i"]] = probs[self.contradiction_idx].item()
                image[2, b["doc_i"], b["gen_i"]] = probs[self.neutral_idx].item()

    def get_cache_file(self):
        """
        Retrieve the path to the cache file.
        """
        return os.path.join(
            self.cache_folder,
            f"cache_{self.model_name}_{self.granularity}.json"
        )

    def save_cache(self):
        """
        Saves the current cache to a file.
        """
        cache_cp = {"[///]".join(k): v.tolist() for k, v in self.cache.items()}
        with open(self.get_cache_file(), "w", encoding="utf-8") as f:
            json.dump(cache_cp, f)

    def load_cache(self):
        """
        Loads the cache from a file and updates the internal cache attribute.
        """
        cache_file = self.get_cache_file()
        if os.path.isfile(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_cp = json.load(f)
                self.cache = {
                    tuple(k.split("[///]")): np.array(v)
                    for k, v in cache_cp.items()
                }


class SummaCConv(torch.nn.Module):
    """
    A class that represents a neural network module for text classification.
    """
    def __init__(
        self,
        config=None,
        models=None,
        start_file=None,
    ):
        if config is None:
            config = {}
        if models is None:
            models = ["mnli", "anli", "vitc"]
        self.models = models
        self.bins = config.get('bins', "even50")
        self.granularity = config.get('granularity', "sentence")
        self.nli_labels = config.get('nli_labels', "e")
        self.device = config.get('device', "cuda")
        self.imager_load_cache = config.get('imager_load_cache', True)
        self.agg = config.get('agg', "mean")
        self.norm_histo = config.get('norm_histo', False)

        assert self.nli_labels in [
            "e",
            "c",
            "n",
            "ec",
            "en",
            "cn",
            "ecn",
        ], f"Unrecognized nli_labels argument {self.nli_labels}"

        super().__init__()
        self.device = self.device
        self.models = models

        self.imagers = []
        for model_name in models:
            self.imagers.append(
                SummaCImager(
                    model_name=model_name, granularity=self.granularity, **config
                )
            )
        if self.imager_load_cache:
            for imager in self.imagers:
                imager.load_cache()
        assert len(self.imagers) > 0, "Imager names were empty or unrecognized"

        if "even" in self.bins:
            n_bins = int(self.bins.replace("even", ""))
            self.bins = list(np.arange(0, 1, 1 / n_bins)) + [1.0]
        elif self.bins == "percentile":
            self.bins = [
                0.0,
                0.01,
                0.02,
                0.03,
                0.04,
                0.07,
                0.13,
                0.37,
                0.90,
                0.91,
                0.92,
                0.93,
                0.94,
                0.95,
                0.955,
                0.96,
                0.965,
                0.97,
                0.975,
                0.98,
                0.985,
                0.99,
                0.995,
                1.0,
            ]

        self.n_bins = len(self.bins) - 1
        self.n_rows = 10
        self.n_labels = 2
        self.n_depth = len(self.imagers) * len(self.nli_labels)
        self.full_size = self.n_depth * self.n_bins
        if self.norm_histo:
            self.full_size += 2

        self.mlp = torch.nn.Linear(self.full_size, 1).to(self.device)
        self.layer_final = torch.nn.Linear(3, self.n_labels).to(self.device)

        if start_file is not None:
            print(self.load_state_dict(torch.load(start_file)))

    def build_image(self, original, generated):
        """
        Builds an image representation from the original and generated texts.
        """
        images = [
            imager.build_image(original, generated) for imager in self.imagers
        ]
        image = np.concatenate(images, axis=0)
        return image

    def compute_histogram(self, original=None, generated=None, image=None):
        """
        Computes the histogram based on the provided inputs.
        """
        if image is None:
            image = self.build_image(original, generated)

        n_depth, n_ori, n_gen = image.shape

        def process_depth(i_depth):
            depth_mod = i_depth % 3
            label_check = self.nli_labels[depth_mod] if depth_mod < len(self.nli_labels) else ''
            if label_check in ['e', 'c', 'n']:
                return np.histogram(
                    image[i_depth, :, i_gen],
                    range=(0, 1),
                    bins=self.bins,
                    density=self.norm_histo,
                )
            return None

        full_histogram = []
        for i_gen in range(n_gen):
            histos = [h for h in (process_depth(i) for i in range(n_depth)) if h is not None]
            if self.norm_histo:
                histos = [[n_ori, n_gen]] + histos
            histogram_row = np.concatenate(histos)
            full_histogram.append(histogram_row)

        full_histogram += [[0.0] * self.full_size] * (self.n_rows - len(full_histogram))
        full_histogram = np.array(full_histogram[: self.n_rows])
        return image, full_histogram

    def aggregate_features(self, rs):
        """
        Aggregates features from the given dataset or resource.
        """
        if self.agg == "mean":
            return torch.mean(rs).repeat(3).unsqueeze(0)
        if self.agg == "min":
            return torch.min(rs).repeat(3).unsqueeze(0)
        if self.agg == "max":
            return torch.max(rs).repeat(3).unsqueeze(0)
        if self.agg == "all":
            return torch.cat([
                torch.min(rs).unsqueeze(0),
                torch.mean(rs).unsqueeze(0),
                torch.max(rs).unsqueeze(0)
            ])
        return torch.zeros(3)  # Default case

    def forward(self, originals, generateds, images=None):
        """
        Perform a forward pass of the model.
        """
        if images is not None:
            histograms = [self.compute_histogram(image=image)[1] for image in images]
        else:
            images, histograms = zip(*[
                self.compute_histogram(original=original, generated=generated)
                for original, generated in zip(originals, generateds)
            ])

        histograms = torch.FloatTensor(histograms).to(self.device)
        non_zeros = (torch.sum(histograms, dim=-1) != 0.0).long()
        seq_lengths = non_zeros.sum(dim=-1).tolist()

        mlp_outs = self.mlp(histograms).reshape(len(histograms), self.n_rows)
        features = []

        for mlp_out, seq_length in zip(mlp_outs, seq_lengths):
            if seq_length > 0:
                rs = mlp_out[:seq_length]
                features.append(self.aggregate_features(rs).unsqueeze(0))
            else:
                features.append(torch.zeros(1, 3))

        features = torch.cat(features)
        logits = self.layer_final(features)

        histograms_out = (
            histograms.cpu().numpy()
            if hasattr(histograms, 'cpu')
            else np.array(histograms)
        )
        return logits, [histograms_out], images

    def save_imager_cache(self):
        """
        Save the cache for all imagers in the current instance.
        """
        for imager in self.imagers:
            imager.save_cache()

    def score(self, originals, generateds):
        """
        Computes the scores for a batch of original and generated texts.
        """
        with torch.no_grad():
            logits= self.forward(originals, generateds)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            batch_scores = probs[:, 1].tolist()
        return {
            "scores": batch_scores
        }

class SummaCZS:
    """
    A class for handling and processing text with multiple models and operations.
    """
    def __init__(
        self,
        model_name="mnli",
        granularity="paragraph",
        options=None,
        args=None,
        **kwargs,
    ):
        if options is None:
            options = {}
        global model_map
        with open(
            os.path.join(args.config_dir, "summac_model.json"),
            "r", 
            encoding="utf-8"
        ) as f:
            model_map = json.load(f)
        # Set default values for options
        default_options = {
            "op1": "max",
            "op2": "mean",
            "use_ent": True,
            "use_con": True,
            "imager_load_cache": True
        }
        # Update default options with provided options
        default_options.update(options)
        assert default_options["op2"] in ["min", "mean", "max"], "Unrecognized `op2`"
        assert default_options["op1"] in ["max", "mean", "min"], "Unrecognized `op1`"

        self.imager = SummaCImager(
            model_name=model_name,
            granularity=granularity,
            **kwargs,
        )
        if default_options["imager_load_cache"]:
            self.imager.load_cache()
        self.op2 = default_options["op2"]
        self.op1 = default_options["op1"]
        self.use_ent = default_options["use_ent"]
        self.use_con = default_options["use_con"]

    def save_imager_cache(self):
        """
        Saves the cache for the imager.
        This method calls the `save_cache` method on the `imager` object to persist
        any cached data. It is typically used to ensure that the current state of
        the imager's cache is saved to disk.
        """
        self.imager.save_cache()

    def score_one(self, original, generated):
        """
        Scores the similarity between the original and generated texts using a predefined imager.
        Args:
            original (str): The original text.
            generated (str): The generated text to compare with the original.
        Returns:
            dict: A dictionary containing:
                - "score" (float): The final score calculated based on the defined operations.
                - "image" (tuple of np.ndarray): A tuple containing two numpy arrays 
                representing the image features
                for the original and generated texts.
        """
        image = self.imager.build_image(original, generated)

        ent_scores = np.max(image[0], axis=0)
        co_scores = np.max(image[1], axis=0)
        if self.op1 == "mean":
            ent_scores = np.mean(image[0], axis=0)
            co_scores = np.mean(image[1], axis=0)
        elif self.op1 == "min":
            ent_scores = np.min(image[0], axis=0)
            co_scores = np.min(image[1], axis=0)

        if self.use_ent and self.use_con:
            scores = ent_scores - co_scores
        elif self.use_ent:
            scores = ent_scores
        elif self.use_con:
            scores = 1.0 - co_scores
        scores = []
        final_score = np.mean(scores)
        if self.op2 == "min":
            final_score = np.min(scores)
        elif self.op2 == "max":
            final_score = np.max(scores)

        return {"score": final_score, "image": image}
    def score(self, sources, generateds):
        "score"
        output = {"scores": [], "images": []}
        for source, gen in zip(sources, generateds):
            score = self.score_one(source, gen)
            output["scores"].append(score["score"])
            output["images"].append(score["image"])
        return output
