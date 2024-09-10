"""
This module contains utility functions for GPU management and batch processing.
"""

import os
import time
import numpy as np

# Ensure tqdm library is installed in your environment
try:
    import tqdm
except ImportError as exc:
    ERROR_MESSAGE = (
        "The 'tqdm' library is not installed. "
        "Please install it using 'pip install tqdm'."
    )
    raise ImportError(ERROR_MESSAGE) from exc

def get_freer_gpu():
    """
    Retrieves the index of the GPU with the most free memory.

    Returns:
        int: The index of the GPU with the most free memory.
    """
    os.system("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp_smi")
    with open("tmp_smi", "r", encoding='utf-8') as file:
        memory_available = [
            int(x.split()[2]) + 5 * i
            for i, x in enumerate(file.readlines())
        ]
    os.remove("tmp_smi")
    return np.argmax(memory_available)

def any_gpu_with_space(gb_needed):
    """
    Checks if there is any GPU with the required amount of free memory.

    Args:
        gb_needed (float): The amount of GPU memory needed in GB.

    Returns:
        bool: True if any GPU has the required amount of free memory, False otherwise.
    """
    os.system("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp_smi")
    with open("tmp_smi", "r", encoding='utf-8') as file:
        memory_available = [
            float(x.split()[2]) / 1024.0
            for x in file.readlines()
        ]
    os.remove("tmp_smi")
    return any(mem >= gb_needed for mem in memory_available)

def wait_free_gpu(gb_needed):
    """
    Waits until a GPU with the required amount of free memory is available.

    Args:
        gb_needed (float): The amount of GPU memory needed in GB.
    """
    while not any_gpu_with_space(gb_needed):
        time.sleep(30)

def select_freer_gpu():
    """
    Selects the GPU with the most free memory and sets it as the visible device.

    Returns:
        str: The index of the selected GPU.
    """
    freer_gpu = str(get_freer_gpu())
    print(f"Will use GPU: {freer_gpu}")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = freer_gpu
    return freer_gpu

def batcher(iterator, batch_size=16, progress=False):
    """
    Batches an iterator into smaller chunks.

    Args:
        iterator (iterable): The iterable to batch.
        batch_size (int): The size of each batch.
        progress (bool): If True, shows a progress bar.

    Yields:
        list: A batch of items from the iterator.
    """
    if progress:
        iterator = tqdm.tqdm(iterator)

    batch = []
    for elem in iterator:
        batch.append(elem)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # Yield remaining items
        yield batch
