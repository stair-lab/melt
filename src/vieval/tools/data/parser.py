import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

from ..utils.constants import DATA_CONFIG


@dataclass
class DatasetAttr:
    r"""
    Dataset attributes.
    """

    # basic configs
    load_from: Literal["hf_hub", "ms_hub", "file"]
    dataset_name: str
    task: Optional[str] = None
    prompting_strategy: Optional[int] = 0
    subset: Optional[str] = None
    train_split: str = "train"
    test_split: str = "test"
    label: Optional[List] = None
    random: Optional[bool] = False
    folder: Optional[str] = None
    num_samples: Optional[int] = None
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(
        self, key: str, obj: Dict[str, Any] = {}, default: Optional[Any] = None
    ) -> None:
        setattr(self, key, obj.get(key, default))


def get_dataset_list(
    dataset_names: Optional[Sequence[str]], dataset_dir: str
) -> List["DatasetAttr"]:
    r"""
    Gets the attributes of the datasets.
    """
    if dataset_names is None:
        dataset_names = []

    config_path = os.path.join(dataset_dir, DATA_CONFIG)

    try:
        with open(config_path, "r") as f:
            dataset_info = json.load(f)
    except Exception as err:
        if len(dataset_names) != 0:
            raise ValueError("Cannot open {} due to {}.".format(config_path, str(err)))

        dataset_info = None

    dataset_list: List["DatasetAttr"] = []
    for name in dataset_names:
        if name not in dataset_info:
            raise ValueError("Undefined dataset {} in {}.".format(name, DATA_CONFIG))

        has_hf_url = "hf_hub_url" in dataset_info[name]
        has_ms_url = "ms_hub_url" in dataset_info[name]

        if has_hf_url or has_ms_url:
            if (has_ms_url) or (not has_hf_url):
                dataset_attr = DatasetAttr(
                    "ms_hub", dataset_name=dataset_info[name]["ms_hub_url"]
                )
            else:
                dataset_attr = DatasetAttr(
                    "hf_hub", dataset_name=dataset_info[name]["hf_hub_url"]
                )
        else:
            dataset_attr = DatasetAttr(
                "file", dataset_name=dataset_info[name]["file_name"]
            )

        dataset_attr.set_attr("subset", dataset_info[name])
        dataset_attr.set_attr("folder", dataset_info[name])
        dataset_attr.set_attr("task", dataset_info[name])
        dataset_attr.set_attr("prompting_strategy", dataset_info[name], default=0)
        dataset_attr.set_attr("random", dataset_info[name], default=False)
        dataset_attr.set_attr("label", dataset_info[name])
        dataset_attr.set_attr("train_split", dataset_info[name], default="train")
        dataset_attr.set_attr("test_split", dataset_info[name], default="test")
        column_names = [
            "context",
            "query",
            "answer",
            "passages",
            "source",
            "target",
            "options",
            "type_id",
        ]
        if "columns" in dataset_info[name]:
            for column_name in column_names:
                dataset_attr.set_attr(
                    column_name, dataset_info[name]["columns"], default=column_name
                )
        else:
            for column_name in column_names:
                dataset_attr.set_attr(column_name, default=column_name)
        dataset_list.append(dataset_attr)

    return dataset_list
