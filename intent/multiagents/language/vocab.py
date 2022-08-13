import csv
import glob
import json
import os
import re
from collections import Counter, OrderedDict
from difflib import get_close_matches

import numpy as np
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

from data_sources.waymo.annotation.html_visualization import MAX_NUM_SENTENCES, find_value_by_key

# Vocabulary for the synthetic language
# The list of vocab should match the filters in data_sources/augment_protobuf_with_language.py
VOCAB = [
    "Stop",
    "MoveFast",
    "MoveSlow",
    "TurnLeft",
    "TurnRight",
    "SpeedUp",
    "SlowDown",
    "LaneChangeLeft",
    "LaneChangeRight",
    "LaneKeep",
    "Follow",
    "Yield",
]
SPECIALS = ["<bos>", "<eos>", "<pad>"]


def get_synthetic_vocab(max_agents: int):
    """Create vocabulary for synthetic language.

    Parameters
    ----------
    max_agents: int
        Maximum number of agents to consider in the predictor.

    Returns
    -------
    Vocab
        A vocab object which maps tokens to indices.
    """
    tokens = SPECIALS.copy()  # Need to copy to avoid modifying the original list
    tokens.extend(VOCAB)
    tokens.extend([str(i) for i in range(max_agents)])
    vocab = torchtext.vocab.vocab(OrderedDict([(token, 1) for token in tokens]))
    return vocab


def clean_caption(caption: str, word_map: dict):
    """Clean up the annotated captions.

    Parameters
    ----------
    caption: str
        The caption text.
    word_map: dict
        The mapping to fix typos.
    """
    caption = caption.lower()
    caption = caption.replace("agent #", "").replace("agent#", "")
    caption = re.sub("[^\w\s]", " ", caption)  # remove punctuations
    caption = re.sub("(?<=\d)(?=[^\d\s])|(?<=[^\d\s])(?=\d)", " ", caption)  # add a space before and after the numbers
    # Fix typos.
    updated_caption = []
    for word in caption.split():
        if word in word_map:
            updated_caption.append(word_map[word])
        else:
            updated_caption.append(word)
    caption = " ".join(updated_caption)
    # Post-processing.
    caption = re.sub("(ego-(\s)?agents?|egoo?(\s)agents?|ego-vehicles?)", "agent", caption)
    caption = caption.replace("-", " ")
    caption = re.sub("[^\w\s]", " ", caption).replace("  ", " ").strip()  # remove punctuations
    caption = (
        caption.replace("t junction", "t-junction").replace("y junction", "y-junction").replace("u turn", "u-turn")
    )
    return caption


def get_word_count(caption_dir: str, word_map: dict):
    """Compute the word count from collected captions.

    Parameters
    ----------
    caption_dir: str
        The path to the folder that contains the annotated captions.
    word_map: dict
        The mapping to fix typos.
    """
    tokenizer = get_tokenizer("basic_english")
    counter = Counter()
    json_files_list = list(glob.glob(os.path.join(caption_dir, "*.json")))
    for json_file in json_files_list:
        with open(json_file, "rb") as fp:
            responses = json.load(fp)
            for response in responses:
                labels = find_value_by_key(response, "labels")
                for s in range(MAX_NUM_SENTENCES):
                    s_idx = "s{}".format(s)
                    if s_idx not in labels:
                        break
                    caption = clean_caption(labels[s_idx], word_map)
                    counter.update([token for token in tokenizer(caption) if not token.isnumeric()])
    return counter


def get_caption_vocab(caption_dir: str, typo_csv: str, max_agents: int):
    """Build vocabulary from collected captions.

    Parameters
    ----------
    caption_dir: str
        The path to the folder that contains the annotated captions.
    typo_csv: str
        The path to the csv mapping to fix typos.
    max_agents: int
        The maximum number of agents to consider in the predictor.
    """
    word_map = dict()
    with open(typo_csv) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            word_map[row[0]] = row[1]
    counter = get_word_count(caption_dir, word_map)
    tokens = SPECIALS
    tokens.extend(list(counter.keys()))
    tokens.extend([str(i) for i in range(max_agents)])
    vocab = torchtext.vocab.vocab(OrderedDict([(token, 1) for token in tokens]))
    return vocab, word_map


def language_onehot_to_tokens(onehot: np.ndarray, vocab: Vocab) -> list:
    """Convert onehot vectors to language tokens.

    Parameters
    ----------
    onehot: np.ndarray
        The onehot vectors of the token sequences. The dimension is (sequence_length, vocab_size).

    Returns
    -------
    out_token_seq: list
        List of token sequences.
    """
    seq_len, vocab_size = onehot.shape
    tokens = []
    for i in range(seq_len):
        token_idx = np.argmax(onehot[i])
        token = vocab.lookup_token(token_idx)
        if token in ["<pad>"]:
            continue
        elif token == "<eos>":
            tokens.append(token)
            break
        else:
            tokens.append(token)
    return tokens


def create_typo_mapping(caption_dir: str, out_file: str, min_count: int = 100):
    """Generate the csv file that contain typo to closest word mapping.

    Parameters
    ----------
    caption_dir: str
        The path to the folder that contains the annotated captions.
    out_file: str
        Output csv file path.
    min_count: int
        The minimum word count to consider as common words.
    """
    existing_corrections = set()
    with open(out_file, "r") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        for row in csv_reader:
            existing_corrections.add(row[0])
    counter = get_word_count(caption_dir, {})
    common_words = set()
    fixes = []
    for word, count in counter.items():
        if count >= min_count:
            common_words.add(word)
        elif word not in existing_corrections:
            matches = get_close_matches(word, common_words)
            if len(matches) > 0:
                fixes.append([word, matches[0]])
    with open(out_file, "a") as csvfile:
        writer = csv.writer(csvfile)
        for fix in fixes:
            writer.writerow(fix)
