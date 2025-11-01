import os

import numpy as np
import urllib.request
from brown_clustering import BigramCorpus, BrownClustering


def download_wikitext2(data_dir="./data"):
    """Download WikiText-2 dataset if not already present"""
    os.makedirs(data_dir, exist_ok=True)

    url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
    valid_url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
    test_url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"

    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "wiki.train.txt")
    valid_path = os.path.join(data_dir, "wiki.valid.txt")
    test_path = os.path.join(data_dir, "wiki.test.txt")

    if not os.path.exists(train_path):
        print("Downloading WikiText-2 train set...")
        urllib.request.urlretrieve(url, train_path)

        print("Downloading WikiText-2 valid set...")
        urllib.request.urlretrieve(valid_url, valid_path)

        print("Downloading WikiText-2 test set...")
        urllib.request.urlretrieve(test_url, test_path)

        print("Download complete!")
    else:
        print("WikiText-2 already downloaded.")

    return data_dir


# def read_file(filepath):
#     """Read and preprocess WikiText file"""
#     with open(filepath, "r", encoding="utf-8") as f:
#         lines = f.readlines()
#
#     paragraphs = []
#     curr_sentences = []
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#
#         if line.startswith("="):
#             curr_paragraph = " ".join(curr_sentences)
#             if curr_paragraph:
#                 paragraphs.append(curr_paragraph)
#             curr_sentences = []
#             continue
#
#         curr_sentences.append(line)
#
#     curr_paragraph = " ".join(curr_sentences)
#     if curr_paragraph:
#         paragraphs.append(curr_paragraph)
#
#     return paragraphs


def read_file(filepath):
    """Read and preprocess WikiText file"""
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Filter out empty lines and headers
    sentences = []
    for line in lines:
        line = line.strip()
        # Skip empty lines and wiki headers
        if line and not line.startswith("="):
            sentences.append(line)

    return sentences


def tokenize(sentences):
    sentences = [sentence.split() + ["<eos>"] for sentence in sentences]
    corpus = BigramCorpus(sentences, alpha=0.5, min_count=0)
    clustering = BrownClustering(corpus, m=128)
    clusters = clustering.train()
    return clusters


def encode_paragraphs(paragraphs, token2idx):
    """
    Encode sentences to token indices

    Args:
        paragraphs: List of sentence strings
        token2idx: Dictionary mapping tokens to indices

    Returns:
        all_tokens: 1D numpy array of all tokens concatenated
        sentence_lengths: 1D numpy array of length of each sentence
    """
    all_tokens = []
    sentence_lengths = []

    unk_idx = token2idx["<unk>"]
    for paragraph in paragraphs:
        tokens = paragraph.split() + ["<eos>"]
        indices = [token2idx.get(token, unk_idx) for token in tokens]

        all_tokens.extend(indices)
        sentence_lengths.append(len(indices))

    return np.array(all_tokens, dtype=np.int32), np.array(
        sentence_lengths, dtype=np.int32
    )


def build_vocab(texts):
    """
    Build vocabulary from list of texts

    Args:
        texts: List of text strings
    """
    print("Building vocabulary...")
    clusters = tokenize(texts)

    token2idx = {}
    token2cluster = {}
    counter = 0
    for cluster, words in enumerate(clusters):
        for word in words:
            token2idx[word] = counter
            token2cluster[counter] = cluster
            counter += 1

    token2cluster_arr = np.zeros(len(token2idx), dtype=np.int32)
    for token, idx in token2idx.items():
        token2cluster_arr[idx] = token2cluster[idx]

    idx2token = {idx: token for token, idx in token2idx.items()}
    return token2idx, idx2token, token2cluster_arr


def save_dataset(dataset, save_dir="./processed_data_brown"):
    """Save processed dataset to disk"""
    os.makedirs(save_dir, exist_ok=True)

    # Save data splits
    np.save(os.path.join(save_dir, "train_tokens.npy"), dataset["train"]["tokens"])
    np.save(os.path.join(save_dir, "train_lengths.npy"), dataset["train"]["lengths"])

    np.save(os.path.join(save_dir, "valid_tokens.npy"), dataset["valid"]["tokens"])
    np.save(os.path.join(save_dir, "valid_lengths.npy"), dataset["valid"]["lengths"])

    np.save(os.path.join(save_dir, "test_tokens.npy"), dataset["test"]["tokens"])
    np.save(os.path.join(save_dir, "test_lengths.npy"), dataset["test"]["lengths"])

    # Save vocabulary
    np.save(os.path.join(save_dir, "token2idx.npy"), dataset["vocab"]["token2idx"])
    np.save(os.path.join(save_dir, "idx2token.npy"), dataset["vocab"]["idx2token"])
    np.save(
        os.path.join(save_dir, "token2cluster_arr.npy"),
        dataset["vocab"]["token2cluster_arr"],
    )

    print(f"\nDataset saved to {save_dir}/")


def load_dataset(save_dir="./processed_data_brown"):
    """Load processed dataset from disk"""
    dataset = {
        "train": {
            "tokens": np.load(os.path.join(save_dir, "train_tokens.npy")),
            "lengths": np.load(os.path.join(save_dir, "train_lengths.npy")),
        },
        "valid": {
            "tokens": np.load(os.path.join(save_dir, "valid_tokens.npy")),
            "lengths": np.load(os.path.join(save_dir, "valid_lengths.npy")),
        },
        "test": {
            "tokens": np.load(os.path.join(save_dir, "test_tokens.npy")),
            "lengths": np.load(os.path.join(save_dir, "test_lengths.npy")),
        },
        "vocab": {
            "token2idx": np.load(
                os.path.join(save_dir, "token2idx.npy"), allow_pickle=True
            ).item(),
            "idx2token": np.load(
                os.path.join(save_dir, "idx2token.npy"), allow_pickle=True
            ).item(),
            "token2cluster_arr": np.load(
                os.path.join(save_dir, "token2cluster_arr.npy"), allow_pickle=True
            ),
        },
    }

    return dataset


if __name__ == "__main__":
    extract_path = download_wikitext2()

    # Read files
    train_path = os.path.join(extract_path, "wiki.train.txt")
    valid_path = os.path.join(extract_path, "wiki.valid.txt")
    test_path = os.path.join(extract_path, "wiki.test.txt")

    print("Reading files...")
    train_paragraphs = read_file(train_path)
    valid_paragraphs = read_file(valid_path)
    test_paragraphs = read_file(test_path)

    print(f"Train paragraphs: {len(train_paragraphs)}")
    print(f"Valid paragraphs: {len(valid_paragraphs)}")
    print(f"Test paragraphs: {len(test_paragraphs)}")

    # Build vocabulary from training data only
    token2idx, idx2token, token2cluster_arr = build_vocab(train_paragraphs)

    # Encode all splits
    print("\nEncoding train set...")
    train_tokens, train_lengths = encode_paragraphs(train_paragraphs, token2idx)

    print("Encoding validation set...")
    valid_tokens, valid_lengths = encode_paragraphs(valid_paragraphs, token2idx)

    print("Encoding test set...")
    test_tokens, test_lengths = encode_paragraphs(test_paragraphs, token2idx)

    # Verify
    assert train_tokens.shape[0] == train_lengths.sum(), "Train: Token count mismatch!"
    assert valid_tokens.shape[0] == valid_lengths.sum(), "Valid: Token count mismatch!"
    assert test_tokens.shape[0] == test_lengths.sum(), "Test: Token count mismatch!"

    print("\n=== Dataset Statistics ===")
    print(f"Train: {len(train_tokens):,} tokens, {len(train_lengths):,} sentences")
    print(f"Valid: {len(valid_tokens):,} tokens, {len(valid_lengths):,} sentences")
    print(f"Test: {len(test_tokens):,} tokens, {len(test_lengths):,} sentences")
    print(f"Vocabulary size: {len(token2idx):,}")

    save_dataset(
        {
            "train": {"tokens": train_tokens, "lengths": train_lengths},
            "valid": {"tokens": valid_tokens, "lengths": valid_lengths},
            "test": {"tokens": test_tokens, "lengths": test_lengths},
            "vocab": {
                "token2idx": token2idx,
                "idx2token": idx2token,
                "token2cluster_arr": token2cluster_arr,
            },
        }
    )
