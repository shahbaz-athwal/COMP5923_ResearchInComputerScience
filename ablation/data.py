"""
Loading the prepared token ids and cutting them into training batches.

A batch is a set of random windows from the text. The target for each window is the
same window shifted one character to the right, so the model learns to predict the
next character at every position at once.
"""

import json
import os

import numpy as np
import torch


def load_split(dataset_directory, split_name):
    # memmap reads the file lazily instead of loading it all into memory, like nanoGPT does.
    return np.memmap(os.path.join(dataset_directory, f"{split_name}.bin"), dtype=np.uint16, mode="r")


def load_metadata(dataset_directory):
    with open(os.path.join(dataset_directory, "meta.json")) as metadata_file:
        return json.load(metadata_file)


def windows_to_tensors(token_ids, start_positions, context_length):
    inputs = np.stack([token_ids[start : start + context_length] for start in start_positions])
    targets = np.stack([token_ids[start + 1 : start + 1 + context_length] for start in start_positions])
    return torch.from_numpy(inputs.astype(np.int64)), torch.from_numpy(targets.astype(np.int64))


def random_batch(token_ids, random_generator, batch_size, context_length, device):
    start_positions = random_generator.integers(0, len(token_ids) - context_length - 1, size=batch_size)
    inputs, targets = windows_to_tensors(token_ids, start_positions, context_length)
    return inputs.to(device), targets.to(device)


def fixed_batches(token_ids, seed, number_of_batches, batch_size, context_length):
    """
    The same batches for every run, used for the loss curves during training. Using a
    fixed seed means two runs are always compared on exactly the same text.
    They stay on the CPU and are moved to the GPU when they are used.
    """
    random_generator = np.random.default_rng(seed)
    return [
        random_batch(token_ids, random_generator, batch_size, context_length, "cpu")
        for _ in range(number_of_batches)
    ]


def every_window(token_ids, context_length, batch_size):
    """
    Non-overlapping windows covering the whole split, for the final evaluation.
    No randomness here, so the final number doesn't depend on which batches got sampled.
    """
    number_of_windows = (len(token_ids) - 1) // context_length
    all_start_positions = np.arange(number_of_windows) * context_length
    for first in range(0, number_of_windows, batch_size):
        start_positions = all_start_positions[first : first + batch_size]
        yield windows_to_tensors(token_ids, start_positions, context_length)
