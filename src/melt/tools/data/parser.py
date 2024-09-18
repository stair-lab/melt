"parser"
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence
from melt.tools.utils.constants import DATA_CONFIG
@dataclass
class SplitConfig:
    "class"
    train: str = "train"
    test: str = "test"
@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    subset: Optional[str] = None
    folder: Optional[str] = None
    task: Optional[str] = None
    label: Optional[List] = None
    splits: SplitConfig = field(default_factory=SplitConfig)
    prompting_strategy: int = 0
    sampling: Dict[str, Any] = field(default_factory=lambda: {"random": False, "num_samples": None})
@dataclass
class DatasetAttr:
    """Dataset attributes."""
    load_from: Literal["hf_hub", "ms_hub", "file"]
    dataset_name: str
    config: DatasetConfig = field(default_factory=DatasetConfig)
    columns: Dict[str, str] = field(default_factory=lambda: {
        "query": "input",
        "response": "output",
        "history": None,
        "context": "context",
        "answer": "answer",
        "passages": "passages",
        "source": "source",
        "target": "target",
        "options": "options",
        "type_id": "type_id"
    })
    def __repr__(self) -> str:
        return self.dataset_name
def load_dataset_config(config_path: str) -> Dict[str, Any]:
    "function"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Config file not found: {config_path}") from err
    except json.JSONDecodeError as err:
        raise ValueError(f"Invalid JSON in config file: {config_path}") from err
def create_dataset_attr(info: Dict[str, Any]) -> DatasetAttr:
    "create"
    if "ms_hub_url" in info or ("hf_hub_url" not in info and "file_name" not in info):
        dataset_attr = DatasetAttr("ms_hub", dataset_name=info.get("ms_hub_url", ""))
    elif "hf_hub_url" in info:
        dataset_attr = DatasetAttr("hf_hub", dataset_name=info["hf_hub_url"])
    else:
        dataset_attr = DatasetAttr("file", dataset_name=info["file_name"])
    config = dataset_attr.config
    config.subset = info.get("subset")
    config.folder = info.get("folder")
    config.task = info.get("task")
    config.label = info.get("label")
    config.prompting_strategy = info.get("prompting_strategy", 0)
    config.splits.train = info.get("train_split", "train")
    config.splits.test = info.get("test_split", "test")
    config.sampling["random"] = info.get("random", False)
    config.sampling["num_samples"] = info.get("num_samples")
    if "columns" in info:
        for column in dataset_attr.columns:
            dataset_attr.columns[column] = info["columns"].get(column, column)
    return dataset_attr
def get_dataset_list(
    dataset_names: Optional[Sequence[str]], dataset_dir: str
) -> List[DatasetAttr]:
    """Gets the attributes of the datasets."""
    if not dataset_names:
        return []
    config_path = os.path.join(dataset_dir, DATA_CONFIG)
    dataset_info = load_dataset_config(config_path)
    dataset_list = []
    for name in dataset_names:
        if name not in dataset_info:
            raise ValueError(f"Undefined dataset {name} in {DATA_CONFIG}")
        dataset_list.append(create_dataset_attr(dataset_info[name]))
    return dataset_list
