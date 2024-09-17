"""Module for loading datasets from various sources."""

import os
from pathlib import Path
from typing import Tuple, Any

# Third-party imports
try:
    from transformers.utils.versions import require_version
except ImportError:
    require_version = None

try:
    from modelscope import MsDataset
    from modelscope.utils.config_ds import MS_DATASETS_CACHE
except ImportError:
    MsDataset = None
    MS_DATASETS_CACHE = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# First-party imports
try:
    from melt.utils.constants import FILEEXT2TYPE
except ImportError:
    FILEEXT2TYPE = {}

def _load_single_dataset(dataset_attr, args, mode) -> Tuple[Any, Any]:
    """
    Load a single dataset based on the given attributes and mode.

    Args:
        dataset_attr: Attributes of the dataset to load.
        args: Arguments containing configuration options.
        mode: The mode of the dataset (e.g., 'train', 'test').

    Returns:
        A tuple containing the loaded dataset and its attributes.

    Raises:
        NotImplementedError: If the load type is unknown.
        ImportError: If required modules are not available.
    """
    print(f"Loading {mode} dataset {dataset_attr}...")

    load_functions = {
        "hf_hub": _load_from_hf_hub,
        "ms_hub": _load_from_ms_hub,
        "file": _load_from_file
    }

    load_func = load_functions.get(dataset_attr.load_from)
    if not load_func:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    return load_func(dataset_attr, args, mode)

def _load_from_hf_hub(dataset_attr, args, mode):
    if load_dataset is None:
        raise ImportError("The 'datasets' library is not installed.")
    return load_dataset(
        path=dataset_attr.dataset_name,
        name=dataset_attr.subset,
        data_dir=dataset_attr.folder,
        split=mode,
        token=args.hf_hub_token,
        trust_remote_code=True,
    ), dataset_attr

def _load_from_ms_hub(dataset_attr, args, mode):
    if MsDataset is None or MS_DATASETS_CACHE is None:
        raise ImportError("ModelScope packages are not installed or not available.")

    if require_version is None:
        raise ImportError("The 'transformers' library is not installed.")

    require_version("modelscope>=1.11.0", "To fix: pip install modelscope>=1.11.0")

    dataset = MsDataset.load(
        dataset_name=dataset_attr.dataset_name,
        subset_name=dataset_attr.subset,
        data_dir=dataset_attr.folder,
        split=mode,
        cache_dir=MS_DATASETS_CACHE,
        token=args.ms_hub_token,
    )

    if isinstance(dataset, MsDataset):
        dataset = dataset.to_hf_dataset()

    return dataset, dataset_attr

def _load_from_file(dataset_attr, args, mode):
    local_path = os.path.join(args.dataset_dir, dataset_attr.dataset_name)
    if not os.path.isdir(local_path):
        raise ValueError(f"Directory {local_path} not found.")

    data_files = {}
    data_path = None

    for file_name in os.listdir(local_path):
        if Path(file_name).stem.split("_")[-1] == mode:
            data_files[mode] = os.path.join(local_path, file_name)
            file_ext = file_name.split(".")[-1]
            current_data_path = FILEEXT2TYPE.get(file_ext)

            if data_path is None:
                data_path = current_data_path
            elif data_path != current_data_path:
                raise ValueError("File types should be identical.")

    if not data_files:
        raise ValueError("No appropriate file found.")

    if data_path is None:
        raise ValueError(f"Allowed file types: {', '.join(FILEEXT2TYPE.keys())}.")

    if load_dataset is None:
        raise ImportError("The 'datasets' library is not installed.")

    return load_dataset(
        path=data_path,
        data_files=data_files,
        split=mode,
        token=args.hf_hub_token,
        trust_remote_code=True,
    ), dataset_attr
