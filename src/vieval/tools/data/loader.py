import os
from pathlib import Path
from datasets import load_dataset
from transformers.utils.versions import require_version
from ..utils.constants import FILEEXT2TYPE


def load_a_dataset(dataset_attr, args):
    dataset_training, _ = _load_single_dataset(
        dataset_attr, args, dataset_attr.train_split
    )
    dataset_testing, _ = _load_single_dataset(
        dataset_attr, args, dataset_attr.test_split
    )
    return dataset_training, dataset_testing


def _load_single_dataset(dataset_attr, args, mode):
    print("Loading {} dataset {}...".format(mode, dataset_attr))
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":
        data_files = {}
        local_path = os.path.join(args.dataset_dir, dataset_attr.dataset_name)

        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                if Path(file_name).stem.split("_")[-1] == mode:
                    data_files[mode] = os.path.join(local_path, file_name)
                    if data_path is None:
                        data_path = FILEEXT2TYPE.get(
                            file_name.split(".")[-1], None
                        )
                    elif data_path != FILEEXT2TYPE.get(
                        file_name.split(".")[-1], None
                    ):
                        raise ValueError("File types should be identical.")

            if len(data_files) < 1:
                raise ValueError("File name is not approriate.")
        # elif os.path.isfile(local_path):  # is file
        #     data_files.append(local_path)
        #     data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
        else:
            raise ValueError("File {} not found.".format(local_path))

        if data_path is None:
            raise ValueError(
                "Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys()))
            )
    else:
        raise NotImplementedError(
            "Unknown load type: {}.".format(dataset_attr.load_from)
        )

    if dataset_attr.load_from == "ms_hub":
        require_version(
            "modelscope>=1.11.0", "To fix: pip install modelscope>=1.11.0"
        )
        from modelscope import MsDataset
        from modelscope.utils.config_ds import MS_DATASETS_CACHE

        cache_dir = MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=mode,
            cache_dir=cache_dir,
            token=args.ms_hub_token,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=mode,
            token=args.hf_hub_token,
            trust_remote_code=True,
        )

    return dataset, dataset_attr
