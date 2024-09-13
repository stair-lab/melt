"""
This module contains configuration and constants related to dataset handling.

It includes:
- DATA_CONFIG: The filename for dataset configuration.
- FILEEXT2TYPE: A dictionary mapping file extensions to their corresponding types.

Constants:
- DATA_CONFIG (str): The name of the configuration file for dataset information.
- FILEEXT2TYPE (dict): A mapping from file extensions to file types used for reading datasets.

Usage:
- Update FILEEXT2TYPE as new file formats are supported.
- Refer to DATA_CONFIG to locate the dataset configuration file.
"""
DATA_CONFIG = "dataset_info.json"

FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}
