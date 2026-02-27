#!/usr/bin/env python3
"""
Basic tests for embeddings output. Run with pytest from project root:
pytest -q
"""
import numpy as np
import pandas as pd
from pathlib import Path

def test_embeddings_exist():
    emb_path = Path("outputs/embeddings.npy")
    idx_path = Path("outputs/embeddings_index.csv")
    assert emb_path.exists(), "outputs/embeddings.npy not found"
    assert idx_path.exists(), "outputs/embeddings_index.csv not found"

def test_no_nans_and_shape():
    emb = np.load("outputs/embeddings.npy")
    assert not np.isnan(emb).any(), "NaN in embeddings"
    idx = pd.read_csv("outputs/embeddings_index.csv")
    assert emb.shape[0] == len(idx), "Embeddings rows != index rows"

def test_duplicate_md5_embeddings_equal():
    idx = pd.read_csv("outputs/embeddings_index.csv")
    emb = np.load("outputs/embeddings.npy")
    # If any md5 duplicates exist, their embeddings should be almost equal
    dupes = idx.groupby("md5").filter(lambda x: len(x) > 1)
    if len(dupes) == 0:
        return
    grouped = dupes.groupby("md5")
    for md5, group in grouped:
        indices = group['idx'].tolist()
        ref = emb[indices[0]]
        for i in indices[1:]:
            assert np.allclose(ref, emb[i], atol=1e-4), f"Duplicate md5 {md5} has differing embeddings"