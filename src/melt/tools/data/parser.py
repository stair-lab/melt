"""
Module for parsing and managing dataset attributes and configurations.

This module provides functionality to load dataset configurations from
a JSON file and manage attributes related to datasets.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence

# Assuming this is the correct import path, adjust if necessary
try:
    from melt.utils.constants import DATA_CONFIG
except ImportError:
    DATA_CONFIG = "data_config.json"  # Fallback value

@dataclass
class ColumnGroup:
    """Group of related column attributes."""
    query: str = "input"
    response: str = "output"
    history: Optional[str] = None
    context: str = "context"

@dataclass
class ColumnAttributes:
    """Attributes related to dataset columns."""
    primary: ColumnGroup = field(default_factory=ColumnGroup)
    answer: str = "answer"
    passages: str = "passages"
    source: str = "source"
    target: str = "target"
    options: str = "options"
    type_id: str = "type_id"

@dataclass
class SplitAttributes:
    """Attributes related to dataset splits."""
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class DatasetConfig:
    """Configuration settings for the dataset."""
    task: Optional[str] = None
    prompting_strategy: int = 0
    subset: Optional[str] = None
    label: Optional[List[Any]] = None
    random: bool = False
    folder: Optional[str] = None
    num_samples: Optional[int] = None

@dataclass
class DatasetMeta:
    """Metadata for managing and loading datasets."""
    config: DatasetConfig = field(default_factory=DatasetConfig)
    columns: ColumnAttributes = field(default_factory=ColumnAttributes)
    splits: SplitAttributes = field(default_factory=SplitAttributes)

@dataclass
class DatasetAttr:
    """Dataset attributes for managing and loading datasets."""
    load_from: Literal["hf_hub", "ms_hub", "file"]
    dataset_name: str
    meta: DatasetMeta = field(default_factory=DatasetMeta)
    extra_attributes: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(self, key: str, obj: Dict[str, Any], default: Any = None) -> None:
        """Set attribute value from a dictionary or use default."""
        if hasattr(self.meta, key):
            setattr(self.meta, key, obj.get(key, default))
        else:
            self.extra_attributes[key] = obj.get(key, default)

def get_dataset_list(
    dataset_names: Optional[Sequence[str]], dataset_dir: str
) -> List[DatasetAttr]:
    """
    Get the attributes of the datasets.

    Args:
        dataset_names: Sequence of dataset names to process.
        dataset_dir: Directory containing the dataset configurations.

    Returns:
        List of DatasetAttr objects.

    Raises:
        ValueError: If the config file cannot be opened or a dataset is undefined.
    """
    dataset_names = dataset_names or []
    config_path = os.path.join(dataset_dir, DATA_CONFIG)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    except (IOError, json.JSONDecodeError) as err:
        if dataset_names:
            raise ValueError(
                f"Cannot open or parse {config_path} due to {str(err)}"
            ) from err
        dataset_info = {}

    dataset_list: List[DatasetAttr] = []
    for name in dataset_names:
        if name not in dataset_info:
            raise ValueError(f"Undefined dataset {name} in {DATA_CONFIG}")

        dataset_attr = create_dataset_attr(name, dataset_info[name])
        set_dataset_attributes(dataset_attr, dataset_info[name])
        dataset_list.append(dataset_attr)

    return dataset_list

def create_dataset_attr(name: str, info: Dict[str, Any]) -> DatasetAttr:
    """Create a DatasetAttr object based on the dataset information."""
    load_from = "ms_hub" if "ms_hub_url" in info or "hf_hub_url" not in info else "hf_hub"
    dataset_name = info.get("ms_hub_url", info.get("hf_hub_url", name))
    return DatasetAttr(load_from=load_from, dataset_name=dataset_name)

def set_dataset_attributes(dataset_attr: DatasetAttr, info: Dict[str, Any]) -> None:
    """Set attributes for a DatasetAttr object."""
    config_attributes = [
        'task', 'prompting_strategy', 'subset', 'label', 'random',
        'folder', 'num_samples'
    ]
    for attr in config_attributes:
        dataset_attr.set_attr(attr, info, default=getattr(dataset_attr.meta.config, attr))

    # Set column attributes if present
    if "columns" in info:
        for column in ColumnAttributes.__annotations__.keys():
            dataset_attr.set_attr(
                column,
                info["columns"],
                default=getattr(dataset_attr.meta.columns, column)
            )

    # Set split attributes if present
    if "splits" in info:
        for split in SplitAttributes.__annotations__.keys():
            dataset_attr.set_attr(
                split,
                info["splits"],
                default=getattr(dataset_attr.meta.splits, split)
            )
