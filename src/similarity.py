#!/usr/bin/env python3
"""
Compute pairwise cosine similarity, top-k retrieval, and visualizations.

Saves:
 - outputs/similarity.npy
 - outputs/topk.csv (query_idx, neighbor_idx, score)
 - outputs/heatmap.png
 - outputs/tsne.png
 - outputs/query_<idx>_neighbors.png (image grids for queries)

Usage:
python src/similarity.py --emb outputs/embeddings.npy --index outputs/embeddings_index.csv --out_dir outputs --topk 5
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from PIL import Image
import math

def pairwise_cosine(emb):
    # emb assumed normalized (L2). If not, normalize.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embn = emb / norms
    sim = embn @ embn.T
    sim = np.clip(sim, -1.0, 1.0)
    return sim

def topk_for_all(sim, k=5):
    n = sim.shape[0]
    rows = []
    for i in range(n):
        row = sim[i]
        idx = np.argsort(-row)
        idx = idx[idx != i][:k]
        for rank, j in enumerate(idx):
            rows.append({"query_idx": i, "neighbor_idx": int(j), "score": float(sim[i,j]), "rank": rank+1})
    return pd.DataFrame(rows)

def plot_heatmap(sim, out_path):
    plt.figure(figsize=(10,8))
    sns.heatmap(sim, cmap="vlag", center=0)
    plt.title("Pairwise Cosine Similarity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_tsne(emb, labels, out_path):
    ts = TSNE(n_components=2, random_state=42, metric="cosine", perplexity=30)
    z = ts.fit_transform(emb)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=z[:,0], y=z[:,1], hue=labels, palette="tab10", s=40)
    plt.title("t-SNE of embeddings")
    plt.legend(title="video_id")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def grid_images(paths, out_path, cols=6, thumb_size=(224,224), title=None):
    imgs = [Image.open(p).convert("RGB").resize(thumb_size) for p in paths]
    rows = math.ceil(len(imgs)/cols)
    w, h = thumb_size
    grid = Image.new('RGB', (cols*w, rows*h), color=(255,255,255))
    for idx, im in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        grid.paste(im, (c*w, r*h))
    grid.save(out_path)

def main(args):
    emb = np.load(args.emb)
    idx_df = pd.read_csv(args.index)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = pairwise_cosine(emb)
    np.save(out_dir / "similarity.npy", sim)
    print("Saved similarity matrix:", sim.shape)

    # top-k neighbors
    topk_df = topk_for_all(sim, k=args.topk)
    topk_df.to_csv(out_dir / "topk.csv", index=False)
    print("Saved topk.csv with", len(topk_df), "rows")

    # heatmap
    plot_heatmap(sim, out_dir / "heatmap.png")

    # tsne colored by video_id
    labels = idx_df['video_id'].astype(str).values
    plot_tsne(emb, labels, out_dir / "tsne.png")

    # select 3 query frames: first frame from each video_id if present
    queries = []
    for vid in idx_df['video_id'].unique()[:3]:
        sub = idx_df[idx_df['video_id'] == vid]
        if len(sub) > 0:
            queries.append(int(sub.iloc[0]['idx']))
    if not queries and len(idx_df) > 0:
        queries = [0, min(1, len(idx_df)-1), min(2, len(idx_df)-1)]

    for q in queries:
        neighbors = topk_df[topk_df['query_idx'] == q].sort_values('rank')['neighbor_idx'].tolist()
        # create grid: query first then neighbors
        paths = [idx_df.loc[idx_df['idx'] == q, 'file_path'].values[0]]
        for n in neighbors:
            paths.append(idx_df.loc[idx_df['idx'] == n, 'file_path'].values[0])
        outp = out_dir / f"query_{q}_neighbors.png"
        grid_images(paths, outp, cols=6)
        print("Saved neighbor grid:", outp)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", default="outputs/embeddings.npy")
    parser.add_argument("--index", default="outputs/embeddings_index.csv")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    main(args)