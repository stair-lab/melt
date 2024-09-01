import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
from datasets import load_dataset
from transformers.utils.versions import require_version
from ..utils.constants import FILEEXT2TYPE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_a_dataset(dataset_attr: object, args: object) -> Tuple[Optional[object], Optional[object]]:
    """Load training and testing datasets based on dataset attributes and arguments.

    Args:
        dataset_attr (object): An object containing dataset attributes.
        args (object): An object containing additional arguments.

    Returns:
        Tuple[Optional[object], Optional[object]]: Training and testing datasets.
    """
    logger.info("Starting to load datasets...")

    dataset_training = _load_single_dataset(dataset_attr, args, dataset_attr.train_split)
    dataset_testing = _load_single_dataset(dataset_attr, args, dataset_attr.test_split)

    logger.info("Datasets loaded successfully.")
    return dataset_training, dataset_testing

def _load_single_dataset(dataset_attr: object, args: object, mode: str) -> Optional[object]:
    """Load a single dataset based on the mode (train/test).

    Args:
        dataset_attr (object): An object containing dataset attributes.
        args (object): An object containing additional arguments.
        mode (str): The mode to load ('train' or 'test').

    Returns:
        Optional[object]: The loaded dataset.
    """
    logger.info(f"Loading {mode} dataset for {dataset_attr.dataset_name}...")

    data_path = _get_data_path(dataset_attr, args, mode)
    data_name = dataset_attr.subset
    data_dir = dataset_attr.folder
    data_files = _get_data_files(dataset_attr, args, mode)

    dataset = None

    if dataset_attr.load_from == "ms_hub":
        _check_modelscope_version()
        from modelscope import MsDataset
        from modelscope.utils.config_ds import MS_DATASETS_CACHE

        try:
            dataset = MsDataset.load(
                dataset_name=data_path,
                subset_name=data_name,
                data_dir=data_dir,
                data_files=data_files,
                split=mode,
                cache_dir=MS_DATASETS_CACHE,
                token=args.ms_hub_token,
            )
            if isinstance(dataset, MsDataset):
                dataset = dataset.to_hf_dataset()
        except Exception as e:
            logger.error(f"Failed to load dataset from ModelScope: {e}")
            raise
    elif dataset_attr.load_from == "hf_hub":
        try:
            dataset = load_dataset(
                path=data_path,
                name=data_name,
                data_dir=data_dir,
                data_files=data_files,
                split=mode,
                token=args.hf_hub_token,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.error(f"Failed to load dataset from Hugging Face Hub: {e}")
            raise
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    logger.info(f"{mode.capitalize()} dataset loaded successfully.")
    return dataset

def _get_data_path(dataset_attr: object, args: object, mode: str) -> Optional[str]:
    """Determine the data path based on dataset attributes and mode.

    Args:
        dataset_attr (object): An object containing dataset attributes.
        args (object): An object containing additional arguments.
        mode (str): The mode to determine the data path ('train' or 'test').

    Returns:
        Optional[str]: The determined data path.
    """
    if dataset_attr.load_from in ["hf_hub", "ms_hub"]:
        return dataset_attr.dataset_name

    if dataset_attr.load_from == "file":
        local_path = Path(args.dataset_dir) / dataset_attr.dataset_name
        if not local_path.is_dir():
            raise ValueError(f"Directory {local_path} does not exist.")

        data_path = _determine_data_path(local_path, mode)
        if data_path is None:
            allowed_extensions = ', '.join(FILEEXT2TYPE.keys())
            raise ValueError(f"No valid file types found. Allowed file types: {allowed_extensions}.")
        return data_path

    raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

def _determine_data_path(local_path: Path, mode: str) -> Optional[str]:
    """Determine the appropriate data path from local files based on mode.

    Args:
        local_path (Path): The local directory path.
        mode (str): The mode to determine the data path ('train' or 'test').

    Returns:
        Optional[str]: The determined data path.
    """
    data_path = None
    for file_name in os.listdir(local_path):
        file_path = local_path / file_name
        if file_path.stem.endswith(f"_{mode}"):
            file_ext = file_path.suffix.lstrip('.')
            current_data_path = FILEEXT2TYPE.get(file_ext)

            if current_data_path is None:
                raise ValueError(f"Unsupported file extension: {file_ext}.")

            if data_path is None:
                data_path = current_data_path
            elif data_path != current_data_path:
                raise ValueError("File types should be identical for all files.")

    return data_path

def _get_data_files(dataset_attr: object, args: object, mode: str) -> Dict[str, str]:
    """Get data files for the specified mode.

    Args:
        dataset_attr (object): An object containing dataset attributes.
        args (object): An object containing additional arguments.
        mode (str): The mode to get data files ('train' or 'test').

    Returns:
        Dict[str, str]: A dictionary mapping mode to data file paths.
    """
    data_files = {}
    if dataset_attr.load_from == "file":
        local_path = Path(args.dataset_dir) / dataset_attr.dataset_name
        for file_name in os.listdir(local_path):
            file_path = local_path / file_name
            if file_path.stem.endswith(f"_{mode}"):
                data_files[mode] = str(file_path)

        if not data_files:
            raise ValueError(f"No files found for mode '{mode}' in directory '{local_path}'.")

    return data_files

def _check_modelscope_version():
    """Check and ensure the required ModelScope version is installed."""
    try:
        require_version("modelscope>=1.11.0", "To fix: pip install modelscope>=1.11.0")
    except Exception as e:
        logger.error(f"ModelScope version check failed: {e}")
        raise
