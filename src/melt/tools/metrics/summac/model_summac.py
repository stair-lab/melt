"""
Module for model handling and utility functions for sequence classification.
Source: https://github.com/tingofurro/summac
"""
from typing import Dict, Union, Optional, List
import os
import json
import sys
import numpy as np
import torch

# Import SummaCConvConfig
try:
    from .config import SummaCConvConfig
except ImportError as e:
    print(f"Error importing SummaCConvConfig: {e}", file=sys.stderr)
    print("Ensure 'metrics.summac.config' module is in your Python path.", file=sys.stderr)
    print("Need to add the parent directory of 'metrics' to your PYTHONPATH.", file=sys.stderr)
    SummaCConvConfig = None

# Import transformers
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    print("transformers library is not installed", file=sys.stderr)
    print(" Some functionality may be limited.",file=sys.stderr)
    print("To install, run: pip install transformers", file=sys.stderr)
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

# Import allennlp
try:
    from allennlp.predictors import Predictor
except ImportError:
    print("Warning: 'allennlp' library is not installed.", file=sys.stderr)
    print("To install, run: pip install allennlp", file=sys.stderr)
    Predictor = None

# Import nltk
try:
    import nltk
except ImportError:
    print("Warning: 'nltk' library is not installed. ", file=sys.stderr)
    print("To install, run: pip install nltk", file=sys.stderr)
    nltk = None

# Import utils_misc
try:
    from . import utils_misc
except ImportError as e:
    print(f"Error importing utils_misc: {e}", file=sys.stderr)
    print("Ensure 'utils_misc' module is in the same directory as this script.", file=sys.stderr)
    utils_misc = None

# Check for critical imports
if SummaCConvConfig is None or utils_misc is None:
    print("Critical imports failed.", file=sys.stderr)
    print("Resolve the import issues before using this module.", file=sys.stderr)
    sys.exit(1)

# Rest of your module code goes here

model_map = {}

def card_to_name(card: str) -> str:
    """
    Convert a model card identifier to its corresponding model name.

    Args:
        card (str): The model card identifier.

    Returns:
        str: The name of the model.
    """
    card2name = {v["model_card"]: k for k, v in model_map.items()}
    return card2name.get(card, card)

def name_to_card(name: str) -> str:
    """
    Convert a model name to its corresponding model card identifier.

    Args:
        name (str): The name of the model.

    Returns:
        str: The model card identifier.
    """
    return model_map.get(name, {}).get("model_card", name)

def get_neutral_idx(ent_idx: int, con_idx: int) -> int:
    """
    Get the index of the neutral sentiment (not entity or context).

    Args:
        ent_idx (int): The index of the entity sentiment.
        con_idx (int): The index of the context sentiment.

    R eturns:
        int: The index of the neutral sentiment.
    """
    return list(set([0, 1, 2]) - set([ent_idx, con_idx]))[0]

class SummaCImager:
    """
    A class for creating semantic similarity images between original and generated text.

    Attributes:
        config (dict): Configuration dictionary for model, granularity, caching, etc.
        resources (dict): Dictionary containing model, tokenizer, and other resources.
        cache (dict): Cache for storing precomputed results.
    """

    def __init__(self, **kwargs):
        """
        Initialize the SummaCImager class with configuration.

        Args:
            **kwargs: Configuration parameters including model_name, granularity, use_cache, etc.
        """
        self.config = {
            "model_name": kwargs.get("model_name", "mnli"),
            "granularity": kwargs.get("granularity", "paragraph"),
            "use_cache": kwargs.get("use_cache", True),
            "max_doc_sents": kwargs.get("max_doc_sents", 100),
            "device": kwargs.get("device", "cuda"),
            "cache_folder": kwargs.get("cache_folder", "/export/share/plaban/summac_cache/"),
            "max_input_length": kwargs.get("max_input_length", 500)
        }
        self.resources = {
            "model": None,
            "tokenizer": None
        }
        self.cache = {}
        self.model_card = None  # Added initialization
        self.entailment_idx = None  # Added initialization
        self.contradiction_idx = None  # Added initialization

        # Validate the configuration
        self._validate_config()

    def _validate_config(self):
        """
        Validate the configuration parameters.
        """
        valid_granularities = ["paragraph", "sentence", "document", "2sents", "mixed"]
        granularity = self.config["granularity"]
        grans = granularity.split("-")
        assert all(gran in valid_granularities for gran in grans) and len(grans) <= 2, \
            f"Unrecognized `granularity` {granularity}"
        assert self.config["model_name"] in model_map, \
            f"Unrecognized model name: `{self.config['model_name']}`"

        if self.config["model_name"] != "decomp":
            self.model_card = name_to_card(self.config["model_name"])
            self.entailment_idx = model_map[self.config["model_name"]]["entailment_idx"]
            self.contradiction_idx = model_map[self.config["model_name"]]["contradiction_idx"]
            self.neutral_idx = get_neutral_idx(
                self.entailment_idx, self.contradiction_idx
            )

    def load_nli(self):
        """
        Load the appropriate model for Natural Language Inference (NLI) based on the model name.
        """
        if self.config["model_name"] == "decomp":
            model_url = (
                "https://storage.googleapis.com/allennlp-public-models/"
                "decomposable-attention-elmo-2020.04.09.tar.gz"
            )
            self.resources['model'] = Predictor.from_path(model_url, cuda_device=0)
        else:
            self.resources["tokenizer"] = AutoTokenizer.from_pretrained(self.model_card)
            self.resources["model"] = AutoModelForSequenceClassification.from_pretrained(
                self.model_card
            ).eval()
            self.resources["model"].to(self.config["device"]).half()

    def split_sentences(self, text):
        """
        Split the given text into sentences.

        Args:
            text (str): The text to split into sentences.

        Returns:
            list: A list of sentences.
        """
        sentences = nltk.tokenize.sent_tokenize(text)
        return [sent for sent in sentences if len(sent) > 10]

    def split_2sents(self, text):
        """
        Split the given text into chunks of two sentences each.

        Args:
            text (str): The text to split into two-sentence chunks.

        Returns:
            list: A list of two-sentence chunks.
        """
        sentences = nltk.tokenize.sent_tokenize(text)
        return [
            " ".join(sentences[i:i + 2])
            for i in range(len(sentences) - 1)
        ]

    def split_paragraphs(self, text):
        """
        Split the given text into paragraphs.

        Args:
            text (str): The text to split into paragraphs.

        Returns:
            list: A list of paragraphs.
        """
        if text.count("\n\n") > 0:
            paragraphs = [p.strip() for p in text.split("\n\n")]
        else:
            paragraphs = [p.strip() for p in text.split("\n")]
        return [p for p in paragraphs if len(p) > 10]

    def split_text(self, text):
        """
        Split the text based on the granularity specified in the configuration.

        Args:
            text (str): The text to be split.

        Returns:
            list: A list of text chunks based on the granularity.
        """
        granularity = self.config["granularity"]

        if granularity == "document":
            return [text]
        if granularity == "paragraph":
            return self.split_paragraphs(text)
        if granularity == "sentence":
            return self.split_sentences(text)
        if granularity == "2sents":
            return self.split_2sents(text)
        if granularity == "mixed":
            return (
                self.split_sentences(text) +
                self.split_paragraphs(text)
            )
        raise ValueError(f"Unsupported granularity level: {granularity}")

    def build_image(self, original, generated):
        """
        This function builds a semantic similarity image between original and generated text.
        """
        cache_key = (original, generated)
        if self.config["use_cache"] and cache_key in self.cache:
            cached_image = self.cache[cache_key]
            return cached_image[:, :self.config["max_doc_sents"], :]

        original_chunks = self.split_text(original)
        generated_chunks = self.split_text(generated)

        if self.resources["model"] is None:
            self.load_nli()

        dataset = self.prepare_dataset(original_chunks, generated_chunks)
        image = np.zeros((3, len(original_chunks), len(generated_chunks)))  # Initialize image
        self.process_batches(dataset, image)

        if self.config["use_cache"]:
            self.cache[cache_key] = image

        return image

    def prepare_dataset(self, original_chunks, generated_chunks):
        """
        Prepare the dataset for model inference.

        Args:
        original_chunks (list): List of original text chunks.
        generated_chunks (list): List of generated text chunks.

        Returns:
        list: Dataset ready for inference.
        """
        return [
            {
                "premise": original_chunks[i],
                "hypothesis": generated_chunks[j],
                "doc_i": i,
                "gen_i": j,
            }
            for i in range(len(original_chunks))
            for j in range(len(generated_chunks))
        ]
    def model_inference(self):
        """
            Perform model inference.

            Returns:
            tuple: Lists of entailment, contradiction, and neutral scores.
        """
        # Implement your model inference logic here
        batch_evids = []
        batch_conts = []
        batch_neuts = []
        return batch_evids, batch_conts, batch_neuts

    def process_batches(self, dataset, image):
        """
        Process batches of data and update the image with entailment, 
        contradiction, and neutral scores.

        Args:
            dataset (list): List of data points for model inference.
            image (np.ndarray): The image array to update.
        """
        for batch in utils_misc.batcher(dataset, batch_size=512):
            batch_evids, batch_conts, batch_neuts = self.model_inference()  # No argument passed
            for b, evid, cont, neut in zip(batch, batch_evids, batch_conts, batch_neuts):
                image[0, b["doc_i"], b["gen_i"]] = evid
                image[1, b["doc_i"], b["gen_i"]] = cont
                image[2, b["doc_i"], b["gen_i"]] = neut
    def get_cache_file(self):
        """
        Get the path to the cache file.

        Returns:
            str: The cache file path.
        """
        return os.path.join(
            self.config["cache_folder"],
            f"cache_{self.config['model_name']}_{self.config['granularity']}.json",
        )

    def save_cache(self):
        """
        Save the cache to a file.
        """
        cache_cp = {"[///]".join(k): v.tolist() for k, v in self.cache.items()}
        with open(self.get_cache_file(), "w", encoding="utf-8") as f:
            json.dump(cache_cp, f)

    def load_cache(self):
        """
        Load the cache from a file.
        """
        cache_file = self.get_cache_file()
        if os.path.isfile(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            self.cache = {tuple(k.split("[///]")): np.array(v) for k, v in cache.items()}

class SummaCConv(torch.nn.Module):
    """Compute and process SummaCConv histograms for text evaluation."""

    def __init__(self, config: Dict[str, Union[str, bool, int, None]]):
        """
        Initialize SummaCConv with a configuration dictionary.
        
        :param config: A dictionary containing configuration parameters.
        """
        super().__init__()
        self.config = SummaCConvConfig(config)
        self._validate_nli_labels()

        # Initialize imagers
        self.imagers = [
            SummaCImager(model_name=model_name, **config)
            for model_name in self.config.models
        ]
        if self.config.imager_load_cache:
            for imager in self.imagers:
                imager.load_cache()

        # Define layers
        self.model_config = {
            'n_bins': len(self.config.bins) - 1,
            'n_labels': 2,
            'n_depth': len(self.imagers) * len(self.config.nli_labels),
            'full_size': (len(self.imagers) * len(self.config.nli_labels) * 
            (len(self.config.bins) - 1)+(2 if self.config.norm_histo else 0))
        }
        self.mlp = torch.nn.Linear(self.model_config['full_size'], 1).to(self.config.device)
        self.layer_final = torch.nn.Linear(3, self.model_config['n_labels']).to(self.config.device)

        if self.config.start_file:
            self.load_state_dict(torch.load(self.config.start_file))

    def _validate_nli_labels(self):
        """Validate nli_labels attribute."""
        valid_labels = ["e", "c", "n", "ec", "en", "cn", "ecn"]
        if self.config.nli_labels not in valid_labels:
            raise ValueError(f"Unrecognized nli_labels argument {self.config.nli_labels}")

    def build_image(self, original, generated):
        """Build an image from original and generated texts using the imagers."""
        images = [imager.build_image(original, generated) for imager in self.imagers]
        return np.concatenate(images, axis=0)

    def compute_histogram(self, original=None, generated=None, image=None):
        """Compute histograms from image data."""
        if image is None:
            image = self.build_image(original, generated)

        depth, num_originals, num_generations = image.shape
        full_histogram = []

        for i_gen in range(num_generations):
            histograms = [
                self._compute_depth_histogram(image, i_depth, i_gen)
                for i_depth in range(depth)
            ]

            if self.config.norm_histo:
                histograms = [[num_originals, num_generations]] + histograms
            histogram_row = np.concatenate(histograms)
            full_histogram.append(histogram_row)

        num_rows_missing = self.config.n_rows - len(full_histogram)
        full_histogram.extend([[0.0] * self.model_config['full_size']] * num_rows_missing)
        return np.array(full_histogram[:self.config.n_rows])

    def _compute_depth_histogram(self, image, i_depth, i_gen):
        """Compute histogram for a specific depth and generation."""
        if self._should_compute_histogram(i_depth):
            return np.histogram(
                image[i_depth, :, i_gen],
                range=(0, 1),
                bins=self.config.bins,
                density=self.config.norm_histo
            )[0]
        return np.zeros(self.model_config['n_bins'])

    def _should_compute_histogram(self, i_depth):
        """Determine if histogram should be computed for given depth."""
        label = self.config.nli_labels
        return (
            (i_depth % 3 == 0 and "e" in label) or
            (i_depth % 3 == 1 and "c" in label) or
            (i_depth % 3 == 2 and "n" in label)
        )

    def forward(self, originals, generateds, images=None):
        """Forward pass through the model."""
        histograms = []
        if images is not None:
            if isinstance(images, (list, tuple)):  # Ensure images is iterable
                histograms = [self.compute_histogram(image=image)[1] for image in images]
            else:
                raise ValueError("Expected 'images' to be a list or tuple of images.")
        else:
            images, histograms = zip(*[
                self.compute_histogram(original=original, generated=generated)
                for original, generated in zip(originals, generateds)
            ])
            histograms = list(histograms)  # Ensure histograms is a list

        # Debugging information
        print(f"Type of histograms before processing: {type(histograms)}")
        print(f"Content of histograms before processing: {histograms}")

        # Ensure histograms is a list or tuple
        if not isinstance(histograms, (list, tuple)):
            raise ValueError(f"Expected 'histograms',a list or tuple, got {type(histograms)}.")

        # Convert histograms to tensor
        histograms = torch.FloatTensor(histograms).to(self.config.device)
        non_zeros = (torch.sum(histograms, dim=-1) != 0.0).long()
        seq_lengths = non_zeros.sum(dim=-1).tolist()

        mlp_outs = self.mlp(histograms).reshape(len(histograms), self.config.n_rows)
        features = [
            self._compute_features(mlp_out, seq_length)
            for mlp_out, seq_length in zip(mlp_outs, seq_lengths)
        ]

        features = torch.cat(features)
        logits = self.layer_final(features)

        # Ensure histograms is iterable before using
        histograms_out = []
        if isinstance(histograms, torch.Tensor):
            histograms = histograms.cpu().numpy()
        for histogram in histograms:
            if isinstance(histogram, torch.Tensor):
                histograms_out.append(histogram.cpu().numpy())
            else:
                histograms_out.append(histogram)

        return logits, histograms_out, images

    def _compute_features(self, mlp_out, seq_length):
        """Compute features based on the aggregation method."""
        if seq_length > 0:
            rs = mlp_out[:seq_length]
            feature = self._aggregate_features(rs)
            return torch.cat([feature] * 3).unsqueeze(0)
        return torch.FloatTensor([0.0, 0.0, 0.0]).unsqueeze(0)

    def _aggregate_features(self, rs):
        """Aggregate features based on the aggregation method."""
        if self.config.agg == "mean":
            return torch.mean(rs).unsqueeze(0)
        if self.config.agg == "min":
            return torch.min(rs).unsqueeze(0)
        if self.config.agg == "max":
            return torch.max(rs).unsqueeze(0)
        if self.config.agg == "all":
            return torch.cat([
                torch.min(rs).unsqueeze(0),
                torch.mean(rs).unsqueeze(0),
                torch.max(rs).unsqueeze(0)
            ]).unsqueeze(0)
        return torch.FloatTensor([0.0, 0.0, 0.0]).unsqueeze(0)

    def save_imager_cache(self, imager):
        """Save imager cache if applicable."""
        if self.config.imager_load_cache:
            imager.save_cache()

    def compute_scores(self, originals, generateds):
        """Compute scores based on originals and generated texts."""
        logits, histograms, _ = self(originals, generateds)
        return torch.softmax(logits, dim=-1), histograms


class SummaCZSConfig:
    """
    Configuration class for SummaCZS model.
    """
    model_name: str = "mnli"
    granularity: str = "paragraph"
    op1: str = "max"
    op2: str = "mean"
    use_ent: bool = True
    use_con: bool = True
    imager_load_cache: bool = True
    device: str = "cuda"
    config_dir: Optional[str] = None

    def __init__(self, **kwargs):
        """
        Initialize the SummaCZSConfig with optional overrides.
        
        :param kwargs: Optional keyword arguments to override default values.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{self.__class__.__name__} has no attribute '{key}'")

    def to_dict(self) -> dict:
        """
        Convert the configuration to a dictionary.
        
        :return: Dictionary representation of the configuration.
        """
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }

    def update(self, **kwargs) -> None:
        """
        Update the configuration with new values.
        :param kwargs: Keyword arguments with new values to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:raise AttributeError(f"{self.__class__.__name__}has no attribute '{key}'")
class SummaCZS:
    """
    Class to handle SummaCZS model operations including image generation and scoring.

    Attributes:
        config (SummaCZSConfig): Configuration object with parameters.
    """
    def __init__(self, config: SummaCZSConfig):
        """
        Initialize the SummaCZS class with the given configuration.

        :param config: Configuration object with parameters.
        """
        self.config = config
        self.model_map = self._load_model_map(config.config_dir)
        self._validate_operations(config.op1, config.op2)

        self.imager = SummaCImager(
            model_name=config.model_name,
            granularity=config.granularity,
            device=config.device,
        )
        if config.imager_load_cache:
            self.imager.load_cache()

        self.op2 = config.op2
        self.op1 = config.op1
        self.use_ent = config.use_ent
        self.use_con = config.use_con

    def _load_model_map(self, config_dir: Optional[str]) -> Dict:
        """Load model configuration from a JSON file."""
        if config_dir is None:
            raise ValueError("config_dir must be specified")
        model_map_path = os.path.join(config_dir, "summac_model.json")
        with open(model_map_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _validate_operations(self, op1: str, op2: str):
        """Validate the operations provided for scoring."""
        valid_ops = ["min", "mean", "max"]
        if op1 not in valid_ops:
            raise ValueError(f"Unrecognized `op1`: {op1}. Must be one of {valid_ops}.")
        if op2 not in valid_ops:
            raise ValueError(f"Unrecognized `op2`: {op2}. Must be one of {valid_ops}.")

    def save_imager_cache(self):
        """Save the imager cache."""
        self.imager.save_cache()

    def score_one(self, original: str, generated: str) -> Dict[str, float]:
        """
        Compute the score for a single pair of original and generated text.

        :param original: Original text.
        :param generated: Generated text.
        :return: Dictionary with the score and image.
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
        else:
            scores = np.zeros_like(ent_scores)  # Ensure `scores` is defined if no condition is met

        final_score = np.mean(scores)
        if self.op2 == "min":
            final_score = np.min(scores)
        elif self.op2 == "max":
            final_score = np.max(scores)

        return {"score": final_score, "image": image}

    def score(self, sources: List[str], generateds: List[str]) -> Dict[str, List[float]]:
        """
        Compute scores for multiple pairs of original and generated text.

        :param sources: List of original texts.
        :param generateds: List of generated texts.
        :return: Dictionary with lists of scores and images.
        """
        output = {"scores": [], "images": []}
        for source, gen in zip(sources, generateds):
            score = self.score_one(source, gen)
            output["scores"].append(score["score"])
            output["images"].append(score["image"])
        return output
