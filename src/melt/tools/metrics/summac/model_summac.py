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

class ConfigManager:
    """
    Manages configuration settings for the application.
    """
    def __init__(self, config=None):
        default_config = {
            'bins': "even50",
            'granularity': "sentence",
            'nli_labels': "e",
            'device': "cuda",
            'imager_load_cache': True,
            'agg': "mean",
            'norm_histo': False
        }
        self._settings = default_config.copy()
        if config:
            self._settings.update(config)
        self._validate_nli_labels()

    def _validate_nli_labels(self):
        valid_labels = ["e", "c", "n", "ec", "en", "cn", "ecn"]
        if self.settings['nli_labels'] not in valid_labels:
            raise ValueError(f"Unrecognized nli_labels argument {self.settings['nli_labels']}")

    def get(self, key, default=None):
        """
        Retrieves the value associated with the specified key from the instance.
        """
        return self.settings.get(key, default)

    def __getattr__(self, name):
        """
        Retrieves the value of the attribute named `name` 
        if it exists; otherwise, raises an AttributeError.
        """
        return self.settings.get(name)
class ImagerManager:
    """
    Manages a collection of image processing or imaging components.
    """
    def __init__(self, models, config):
        self.models = models
        self.config = config
        self.imagers = self._create_imagers()

    def _create_imagers(self):
        imagers = []
        for model_name in self.models:
            imager = SummaCImager(
                model_name=model_name,
                granularity=self.config.granularity,
                **self.config.config
            )
            imagers.append(imager)
        if self.config.imager_load_cache:
            for imager in imagers:
                imager.load_cache()
        if not imagers:
            raise ValueError("Imager names were empty or unrecognized")
        return imagers
    def build_image(self, original, generated):
        """
        Builds an image representation based on the original and generated data.
        """
        images = [
            imager.build_image(original, generated) for imager in self.imagers
        ]
        return np.concatenate(images, axis=0)

    def save_imager_cache(self):
        """
        Saves the current state of the imager cache to persistent storage.
        """
        for imager in self.imagers:
            imager.save_cache()
class CacheConfig:
    """
    Configures settings for cache management.
    """
    def __init__(self, use_cache=True, cache_folder="/export/share/plaban/summac_cache/"):
        self.use_cache = use_cache
        self.cache_folder = cache_folder

    def is_cache_enabled(self):
        """
        Checks if caching is enabled in the current configuration.
        """
        return self.use_cache

    def get_cache_folder(self):
        """
        Retrieves the path to the folder where cache files are stored.
        """
        return self.cache_folder
class Config:
    """
    Represents the configuration settings for the application or component.
    """
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.cache_config = CacheConfig(
            use_cache=config.get('use_cache', True),
            cache_folder=config.get('cache_folder', "/export/share/plaban/summac_cache/")
        )
        self.max_doc_sents = config.get('max_doc_sents', 100)
        self.max_input_length = config.get('max_input_length', 500)
        self.device = config.get('device', "cuda")

    def get_max_doc_sents(self):
        """
        Retrieves the maximum number of document sentences allowed by the instance.
        """
        return self.max_doc_sents

    def get_max_input_length(self):
        """
        Retrieves the maximum length of input allowed by the instance.
        """
        return self.max_input_length

    def get_device(self):
        """
        Retrieves the device currently used by the instance for computations.
        """
        return self.device

    def update_config(self, new_config):
        """
        Updates the configuration of the instance with new settings.
        """
        if 'use_cache' in new_config or 'cache_folder' in new_config:
            self.cache_config = CacheConfig(
                use_cache=new_config.get('use_cache', self.cache_config.use_cache),
                cache_folder=new_config.get('cache_folder', self.cache_config.cache_folder)
            )
        self.max_doc_sents = new_config.get('max_doc_sents', self.max_doc_sents)
        self.max_input_length = new_config.get('max_input_length', self.max_input_length)
        self.device = new_config.get('device', self.device)

    def to_dict(self):
        """
        Converts the instance attributes of the class into a dictionary.
        """
        return {
            'use_cache': self.cache_config.use_cache,
            'cache_folder': self.cache_config.cache_folder,
            'max_doc_sents': self.max_doc_sents,
            'max_input_length': self.max_input_length,
            'device': self.device
        }
class TextSplitter:
    """
    Splits text into various chunks based on specified granularity.
    """
    @staticmethod
    def split_sentences(text):
        """
        Splits the given text into individual sentences.
        """
        sentences = nltk.tokenize.sent_tokenize(text)
        return [sent for sent in sentences if len(sent) > 10]

    @staticmethod
    def split_2sents(text):
        """
        Splits the given text into chunks of two sentences each.
        """
        sentences = TextSplitter.split_sentences(text)
        return [" ".join(sentences[i:(i + 2)]) for i in range(len(sentences))]

    @staticmethod
    def split_paragraphs(text):
        """
        Splits the given text into paragraphs.
        """
        if text.count("\n\n") > 0:
            paragraphs = [p.strip() for p in text.split("\n\n")]
        else:
            paragraphs = [p.strip() for p in text.split("\n")]
        return [p for p in paragraphs if len(p) > 10]

    @staticmethod
    def split_text(text, granularity="sentence"):
        """
        Splits the given text into chunks based on the specified granularity.
        """
        if granularity == "document":
            return [text]
        if granularity == "paragraph":
            return TextSplitter.split_paragraphs(text)
        if granularity == "sentence":
            return TextSplitter.split_sentences(text)
        if granularity == "2sents":
            return TextSplitter.split_2sents(text)
        if granularity == "mixed":
            return TextSplitter.split_sentences(text) + TextSplitter.split_paragraphs(text)
        raise ValueError(f"Invalid granularity: {granularity}")
class ModelLoader:
    """
    Loads and manages machine learning models for inference.
    """
    @staticmethod
    def load_nli(model_name, model_card, device):
        """
        Loads the appropriate NLI (Natural Language Inference) model based on the model name.
        """
        if model_name == "decomp":
            return ModelLoader._load_decomp_model()

        return ModelLoader._load_transformer_model(model_card, device)

    @staticmethod
    def _load_decomp_model():
        predictor_module = importlib.import_module('allennlp.predictors.predictor')
        predictor_class = getattr(predictor_module, 'Predictor')
        model = predictor_class.from_path(
            "https://storage.googleapis.com/allennlp-public-models"
            "/decomposable-attention-elmo-2020.04.09.tar.gz",
            cuda_device=0,
        )
        return model, None

    @staticmethod
    def _load_transformer_model(model_card, device):
        tokenizer = AutoTokenizer.from_pretrained(model_card)
        model = AutoModelForSequenceClassification.from_pretrained(model_card).eval()
        model.to(device).half()
        return model, tokenizer

    @staticmethod
    def is_decomp_model(model_name):
        """
        Checks if the given model name corresponds to a decompositional model.
        """
        return model_name == "decomp"

    @staticmethod
    def get_model_type(model_name):
        """
        Determines the type of model based on its name.
        """
        return "decomp" if ModelLoader.is_decomp_model(model_name) else "transformer"

    @staticmethod
    def get_model_info(model_name, model_card):
        """
        Retrieves model information including model name, card, type, 
        and whether a tokenizer is required.
        """
        model_type = ModelLoader.get_model_type(model_name)
        return {
            "model_name": model_name,
            "model_card": model_card,
            "model_type": model_type,
            "requires_tokenizer": model_type != "decomp"
        }
class CacheManager:
    """
    A class to manage caching for model outputs based on specified granularity.
    """
    def __init__(self, cache_folder, model_name, granularity):
        self.cache_folder = cache_folder
        self.model_name = model_name
        self.granularity = granularity
        self.cache = {}

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
class ModelConfig:
    """
    A class to handle configuration for different models.
    """

    def __init__(self, model_name):
        """
        Initializes the ModelConfig instance with the model's name and relevant indices.

        Args:
            model_name (str): The name of the model.
        """
        self.model_name = model_name
        self.model_card = None
        self.entailment_idx = None
        self.contradiction_idx = None
        self.neutral_idx = None
        if self.model_name != "decomp":
            self.model_card = name_to_card(self.model_name)
            self.entailment_idx = model_map[self.model_name]["entailment_idx"]
            self.contradiction_idx = model_map[self.model_name]["contradiction_idx"]
            self.neutral_idx = get_neutral_idx(self.entailment_idx, self.contradiction_idx)

    def get_model_details(self):
        """
        Returns the details of the model configuration.

        Returns:
            dict: A dictionary containing the model's name, card, and indices.
        """
        return {
            "model_name": self.model_name,
            "model_card": self.model_card,
            "entailment_idx": self.entailment_idx,
            "contradiction_idx": self.contradiction_idx,
            "neutral_idx": self.neutral_idx
        }

    def is_valid(self):
        """
        Checks if the model configuration is valid.

        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        is_model_in_map = self.model_name in model_map
        are_indices_set = (self.entailment_idx is not None
                        and self.contradiction_idx is not None)
        return is_model_in_map and are_indices_set

    def __str__(self):
        """
        Returns a string representation of the model configuration.

        Returns:
            str: A string summarizing the model configuration.
        """
        details = self.get_model_details()
        return (f"ModelConfig(model_name={details['model_name']}, "
            f"model_card={details['model_card']}, "
            f"entailment_idx={details['entailment_idx']}, "
            f"contradiction_idx={details['contradiction_idx']}, "
            f"neutral_idx={details['neutral_idx']})")
class SummaCImager:
    """
    A class to handle the imager models for text classification.
    """
    def __init__(self, model_name="mnli", granularity="paragraph", config=None):
        self.config = Config(config)
        self.grans = granularity.split("-")
        self._validate_inputs(model_name, granularity)
        self.model_config = ModelConfig(model_name)
        self.cache_manager = CacheManager(
            self.config.cache_config.cache_folder,
            model_name,
            granularity
        )
        self.text_splitter = TextSplitter()
        self.model_loader = ModelLoader()

    def _validate_inputs(self, model_name, granularity):
        assert all(
            gran in ["paragraph", "sentence", "document", "2sents", "mixed"]
            for gran in self.grans
        ) and len(self.grans) <= 2, (
            f"Unrecognized `granularity` {granularity}"
        )
        # pylint: disable=undefined-variable
        assert model_name in model_map, f"Unrecognized model name: `{model_name}`"

    def load_nli(self):
        """
        Loads the NLI model based on the specified model name.
        """
        return self.model_loader.load_nli(
            self.model_config.model_name,
            self.model_config.model_card,
            self.config.device
        )

    def build_image(self, original, generated):
        """
        Builds an image representation from the original and generated texts.
        """
        cache_key = (original, generated)
        if self._should_use_cache(cache_key):
            return self._get_cached_image(cache_key)
        original_chunks, generated_chunks = self._prepare_chunks(original, generated)
        num_ori, num_gen = len(original_chunks), len(generated_chunks)

        if num_ori == 0 or num_gen == 0:
            return np.zeros((3, 1, 1))
        image = np.zeros((3, num_ori, num_gen))
        model, tokenizer = self.load_nli()

        dataset = self._create_dataset(original_chunks, generated_chunks)

        self._process_dataset(dataset, image, model, tokenizer)

        self._cache_image(cache_key, image)
        return image

    def _should_use_cache(self, cache_key):
        return self.config.cache_config.is_cache_enabled() and cache_key in self.cache_manager.cache

    def _get_cached_image(self, cache_key):
        return self.cache_manager.cache[cache_key][:, :self.config.max_doc_sents, :]

    def _prepare_chunks(self, original, generated):
        gran_doc, gran_sum = self.grans[0], self.grans[-1]
        chunks = self.text_splitter.split_text(
            original,
            granularity=gran_doc
        )
        original_chunks = chunks[:self.config.max_doc_sents]
        generated_chunks = self.text_splitter.split_text(generated, granularity=gran_sum)
        return original_chunks, generated_chunks

    def _create_dataset(self, original_chunks, generated_chunks):
        return [
            {
                "premise": original_chunks[i], 
                "hypothesis": generated_chunks[j], 
                "doc_i": i, 
                "gen_i": j
            }
            for i in range(len(original_chunks)) for j in range(len(generated_chunks))
        ]

    def _process_dataset(self, dataset, image, model, tokenizer):
        for batch in utils_misc.batcher(dataset, batch_size=512):
            self._process_batch(batch, image, model, tokenizer)

    def _cache_image(self, cache_key, image):
        if self.config.cache_config.is_cache_enabled():
            self.cache_manager.cache[cache_key] = image

    def _process_batch(self, batch, image, model, tokenizer):
        if self.model_config.model_name == "decomp":
            self._process_decomp_batch(batch, image, model)
        else:
            self._process_transformer_batch(batch, image, model, tokenizer)

    def _process_decomp_batch(self, batch, image, model):
        batch_json = [{"premise": d["premise"], "hypothesis": d["hypothesis"]} for d in batch]
        model_outs = model.predict_batch_json(batch_json)
        for out, b in zip(model_outs, batch):
            probs = out["label_probs"]
            image[0, b["doc_i"], b["gen_i"]] = probs[0]
            image[1, b["doc_i"], b["gen_i"]] = probs[1]
            image[2, b["doc_i"], b["gen_i"]] = probs[2]

    def _process_transformer_batch(self, batch, image, model, tokenizer):
        batch_prems = [b["premise"] for b in batch]
        batch_hypos = [b["hypothesis"] for b in batch]
        batch_tokens = tokenizer.batch_encode_plus(
            list(zip(batch_prems, batch_hypos)),
            padding=True,
            truncation=True,
            max_length=self.config.max_input_length,
            return_tensors="pt",
            truncation_strategy="only_first",
        )
        batch_tokens = {k: v.to(self.config.device) for k, v in batch_tokens.items()}
        with torch.no_grad():
            model_outputs = model(**batch_tokens)

        batch_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)
        for b, probs in zip(batch, batch_probs):
            image[0, b["doc_i"], b["gen_i"]] = probs[self.model_config.entailment_idx].item()
            image[1, b["doc_i"], b["gen_i"]] = probs[self.model_config.contradiction_idx].item()
            image[2, b["doc_i"], b["gen_i"]] = probs[self.model_config.neutral_idx].item()

    def save_cache(self):
        """
        Saves the current cache to a file.
        """
        self.cache_manager.save_cache()

    def load_cache(self):
        """
        Loads the cache from a file and updates the internal cache attribute.
        """
        self.cache_manager.load_cache()

class HistogramComputer:
    """
    A class to compute histograms based on configuration and depth.
    """
    def __init__(self, config, n_depth):
        self.config = config
        self.n_depth = n_depth
        self.bins = self._setup_bins()
        self.n_bins = len(self.bins) - 1
        self.full_size = self.n_depth * self.n_bins
        if self.config.norm_histo:
            self.full_size += 2

    def _setup_bins(self):
        if "even" in self.config.bins:
            n_bins = int(self.config.bins.replace("even", ""))
            return list(np.arange(0, 1, 1 / n_bins)) + [1.0]
        if self.config.bins == "percentile":
            return [
                0.0, 0.01, 0.02, 0.03, 0.04, 0.07, 0.13, 0.37, 0.90, 0.91,
                0.92, 0.93, 0.94, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975,
                0.98, 0.985, 0.99, 0.995, 1.0,
            ]
        raise ValueError(f"Unrecognized bins configuration: {self.config.bins}")

    def compute_histogram(self, image):
        """
        Computes the histogram based on the provided inputs.
        """
        n_depth, n_ori, n_gen = image.shape

        def process_depth(i_depth):
            depth_mod = i_depth % 3
            label_check = (
                self.config.nli_labels[depth_mod]
                if depth_mod < len(self.config.nli_labels)
                else ''
            )
            if label_check in ['e', 'c', 'n']:
                return np.histogram(
                    image[i_depth, :, i_gen],
                    range=(0, 1),
                    bins=self.bins,
                    density=self.config.norm_histo,
                )
            return None

        full_histogram = []
        for i_gen in range(n_gen):
            histos = [h for h in (process_depth(i) for i in range(n_depth)) if h is not None]
            if self.config.norm_histo:
                histos = [[n_ori, n_gen]] + histos
            histogram_row = np.concatenate([h[0] for h in histos])
            full_histogram.append(histogram_row)

        full_histogram += [[0.0] * self.full_size] * (10 - len(full_histogram))
        full_histogram = np.array(full_histogram[:10])
        return full_histogram

    def get_histogram_info(self):
        """
        Returns information about the histogram configuration.
        
        This method provides a summary of the current histogram settings,
        which can be useful for debugging or logging purposes.
        """
        return {
            "n_depth": self.n_depth,
            "n_bins": self.n_bins,
            "full_size": self.full_size,
            "bin_edges": self.bins,
            "normalization": self.config.norm_histo
        }


class SummaCConv(torch.nn.Module):
    """
    A class that represents a neural network module for text classification.
    """
    def __init__(self, config=None, models=None, start_file=None):
        super().__init__()
        self.config_manager = ConfigManager(config)
        self.imager_manager = ImagerManager(models or ["mnli", "anli", "vitc"], self.config_manager)
        self.histogram_computer = HistogramComputer(
            self.config_manager,
            len(self.imager_manager.imagers) * len(self.config_manager.nli_labels)
        )
        self.mlp = torch.nn.Linear(
            self.histogram_computer.full_size,
            1
        ).to(self.config_manager.device)
        self.layer_final = torch.nn.Linear(3, 2).to(self.config_manager.device)

        if start_file is not None:
            print(self.load_state_dict(torch.load(start_file)))

    def aggregate_features(self, rs):
        """
        Aggregates features from the given dataset or resource.
        """
        if self.config_manager.agg == "mean":
            return torch.mean(rs).repeat(3).unsqueeze(0)
        if self.config_manager.agg == "min":
            return torch.min(rs).repeat(3).unsqueeze(0)
        if self.config_manager.agg == "max":
            return torch.max(rs).repeat(3).unsqueeze(0)
        if self.config_manager.agg == "all":
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
        if images is None:
            images = [self.imager_manager.build_image(original, generated)
                      for original, generated in zip(originals, generateds)]
        histograms = [self.histogram_computer.compute_histogram(image) for image in images]
        histograms = torch.FloatTensor(histograms).to(self.config_manager.device)
        non_zeros = (torch.sum(histograms, dim=-1) != 0.0).long()
        seq_lengths = non_zeros.sum(dim=-1).tolist()

        mlp_outs = self.mlp(histograms).reshape(len(histograms), 10)
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
        Saves the cache for the imager.
        """
        self.imager_manager.save_imager_cache()

    def score(self, originals, generateds):
        "score"
        with torch.no_grad():
            logits, _, _ = self.forward(originals, generateds)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            batch_scores = probs[:, 1].tolist()
        return {"scores": batch_scores}

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
        self.model_map = self._load_model_map(args)
        default_options = {
            "op1": "max",
            "op2": "mean",
            "use_ent": True,
            "use_con": True,
            "imager_load_cache": True
        }
        default_options.update(options)
        assert default_options["op2"] in ["min", "mean", "max"], "Không nhận ra `op2`"
        assert default_options["op1"] in ["max", "mean", "min"], "Không nhận ra `op1`"

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

    def _load_model_map(self, args):
        with open(
            os.path.join(args.config_dir, "summac_model.json"),
            "r", 
            encoding="utf-8"
        ) as f:
            return json.load(f)

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
