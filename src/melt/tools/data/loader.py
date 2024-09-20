"Loader"
import os
from pathlib import Path
from transformers.utils.versions import require_version
from modelscope import MsDataset
from modelscope.utils.config_ds import MS_DATASETS_CACHE
from datasets import load_dataset
from melt.tools.utils.constants import FILEEXT2TYPE

def load_a_dataset(dataset_attr, args):
    """Load dataset for training and testing"""
    dataset_training, _ = _load_single_dataset(
        dataset_attr, args, dataset_attr.train_split
    )
    dataset_testing, _ = _load_single_dataset(
        dataset_attr, args, dataset_attr.test_split
    )
    return dataset_training, dataset_testing

def _load_single_dataset(dataset_attr, args, mode):
    print(f"Loading {mode} dataset {dataset_attr}...")
    load_config = _get_load_config(dataset_attr, args, mode)
    if dataset_attr.load_from == "ms_hub":
        dataset = _load_from_ms_hub(load_config, args, mode)
    else:
        dataset = _load_from_hf_hub(load_config, args, mode)
    return dataset, dataset_attr
def _get_load_config(dataset_attr, args, mode):
    config = {
        "data_path": None,
        "data_name": None,
        "data_dir": None,
        "data_files": None,
    }
    if dataset_attr.load_from in ["hf_hub", "ms_hub"]:
        config["data_path"] = dataset_attr.dataset_name
        config["data_name"] = dataset_attr.subset
        config["data_dir"] = dataset_attr.folder
    elif dataset_attr.load_from == "file":
        config["data_files"], config["data_path"] = _get_file_config(dataset_attr, args, mode)
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")
    return config
def _get_file_config(dataset_attr, args, mode):
    local_path = os.path.join(args.dataset_dir, dataset_attr.dataset_name)
    if not os.path.isdir(local_path):
        raise ValueError(f"Directory {local_path} not found.")
    data_files = {}
    data_path = None
    for file_name in os.listdir(local_path):
        if Path(file_name).stem.split("_")[-1] == mode:
            data_files[mode] = os.path.join(local_path, file_name)
            file_type = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
            if data_path is None:
                data_path = file_type
            elif data_path != file_type:
                raise ValueError("File types should be identical.")
    if not data_files:
        raise ValueError("No matching files found.")
    if data_path is None:
        raise ValueError(f"Unable to determine file type for {local_path}.")
    return data_files, data_path
def _load_from_ms_hub(config, args, mode):
    require_version("modelscope>=1.11.0", "To fix: pip install modelscope>=1.11.0")
    dataset = MsDataset.load(
        dataset_name=config["data_path"],
        subset_name=config["data_name"],
        data_dir=config["data_dir"],
        data_files=config["data_files"],
        split=mode,
        cache_dir=MS_DATASETS_CACHE,
        token=args.ms_hub_token,
    )
    return dataset.to_hf_dataset() if isinstance(dataset, MsDataset) else dataset
def _load_from_hf_hub(config, args, mode):
    return load_dataset(
        path=config["data_path"],
        name=config["data_name"],
        data_dir=config["data_dir"],
        data_files=config["data_files"],
        split=mode,
        token=args.hf_hub_token,
        trust_remote_code=True,
    )
