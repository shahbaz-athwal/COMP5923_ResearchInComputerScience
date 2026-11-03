"""
Prepare the tiny Shakespeare dataset at the character level.

This follows nanoGPT's data/shakespeare_char/prepare.py: every distinct character
becomes one token, the first 90% of the text is used for training and the last 10%
for validation. It writes three files next to this script:

    train.bin   token ids of the training text (uint16)
    val.bin     token ids of the validation text (uint16)
    meta.json   vocabulary size and the character <-> id lookup tables

    python data/shakespeare_char/prepare.py
"""

import json
import os
import urllib.request

import numpy as np

SOURCE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATASET_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def download_text():
    text_path = os.path.join(DATASET_DIRECTORY, "input.txt")
    if not os.path.exists(text_path):
        print(f"downloading {SOURCE_URL}")
        urllib.request.urlretrieve(SOURCE_URL, text_path)
    with open(text_path) as text_file:
        return text_file.read()


def main():
    text = download_text()
    print(f"length of dataset in characters: {len(text):,}")

    # The vocabulary is simply every character that appears in the text, in sorted order.
    characters = sorted(set(text))
    character_to_id = {character: index for index, character in enumerate(characters)}
    id_to_character = {index: character for index, character in enumerate(characters)}
    print(f"vocabulary size: {len(characters)}")

    # Contiguous split, so the validation text is a part of the plays the model never sees.
    split_point = int(len(text) * 0.9)
    train_ids = np.array([character_to_id[character] for character in text[:split_point]], dtype=np.uint16)
    validation_ids = np.array([character_to_id[character] for character in text[split_point:]], dtype=np.uint16)
    print(f"train has {len(train_ids):,} tokens, validation has {len(validation_ids):,} tokens")

    train_ids.tofile(os.path.join(DATASET_DIRECTORY, "train.bin"))
    validation_ids.tofile(os.path.join(DATASET_DIRECTORY, "val.bin"))
    metadata = {
        "vocabulary_size": len(characters),
        "character_to_id": character_to_id,
        "id_to_character": id_to_character,
    }
    with open(os.path.join(DATASET_DIRECTORY, "meta.json"), "w") as metadata_file:
        json.dump(metadata, metadata_file)


if __name__ == "__main__":
    main()
