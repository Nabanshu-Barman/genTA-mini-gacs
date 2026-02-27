#!/usr/bin/env python3
"""
Robust CLIP embedding computation with graceful handling of different HF output types.

Usage:
python src/compute_embeddings.py --index data/frames/index_all.csv --out_dir outputs --batch_size 8
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm
import sys

def load_image(path):
    return Image.open(path).convert("RGB")

def extract_tensor_from_model_output(output):
    # If HF returns a Tensor directly
    if torch.is_tensor(output):
        return output
    # Common HF model output attributes to check
    for attr in ("image_embeds", "pooler_output", "last_hidden_state", "hidden_states"):
        val = getattr(output, attr, None)
        if val is not None:
            # last_hidden_state: mean pool over sequence dim (dim=1)
            if attr == "last_hidden_state":
                return val.mean(dim=1)
            # hidden_states: take last element then mean pool (if structure)
            if attr == "hidden_states":
                if isinstance(val, (list, tuple)):
                    last = val[-1]
                    return last.mean(dim=1)
                else:
                    return val.mean(dim=1)
            return val
    # Some models return a dict-like object
    try:
        if isinstance(output, dict):
            for key in ("image_embeds", "pooler_output", "last_hidden_state"):
                if key in output:
                    v = output[key]
                    if isinstance(v, torch.Tensor):
                        if key == "last_hidden_state":
                            return v.mean(dim=1)
                        return v
    except Exception:
        pass
    raise RuntimeError("Could not extract image tensor from model output. Inspect model output keys/attrs.")

def compute_embeddings(file_paths, model_name="openai/clip-vit-base-patch32", batch_size=16, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    all_embs = []
    for i in tqdm(range(0, len(file_paths), batch_size), desc="Embedding batches"):
        batch_paths = file_paths[i:i+batch_size]
        images = [load_image(p) for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        # move tensors to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.get_image_features(**inputs) if hasattr(model, "get_image_features") else model(**inputs)
            feats = extract_tensor_from_model_output(out)
            # ensure tensor shape [B, D]
            if feats.dim() == 1:
                feats = feats.unsqueeze(0)
            # Convert to float32 and L2 normalize
            feats = feats.float()
            norms = feats.norm(p=2, dim=-1, keepdim=True)
            norms[norms == 0] = 1.0
            feats = feats / norms
            all_embs.append(feats.cpu().numpy().astype(np.float32))
    embeddings = np.vstack(all_embs) if all_embs else np.zeros((0, model.config.projection_dim if hasattr(model.config, "projection_dim") else model.config.hidden_size), dtype=np.float32)
    return embeddings

def main(args):
    idx_csv = Path(args.index)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(idx_csv)
    file_paths = df['file_path'].tolist()
    print(f"Found {len(file_paths)} frames in {idx_csv}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    emb = compute_embeddings(file_paths, model_name=args.model, batch_size=args.batch_size, device=device)
    print("Embeddings computed. Shape:", emb.shape)

    # Sanity checks
    if np.isnan(emb).any():
        print("ERROR: NaNs present in embeddings", file=sys.stderr)
    # Save embeddings and index
    np.save(out_dir / "embeddings.npy", emb)
    df_out = df[['video_id','file_path','timestamp_s','md5']].copy()
    df_out.insert(0, "idx", range(len(df_out)))
    df_out.to_csv(out_dir / "embeddings_index.csv", index=False)

    # Save stats
    with open(out_dir / "embedding_stats.txt", "w") as f:
        f.write(f"n_frames: {emb.shape[0]}\n")
        f.write(f"embedding_dim: {emb.shape[1] if emb.shape[0]>0 else 0}\n")
        if emb.shape[0] > 0:
            f.write(f"min: {float(np.min(emb))}\n")
            f.write(f"max: {float(np.max(emb))}\n")
            f.write(f"mean: {float(np.mean(emb))}\n")
    print("Saved embeddings and index to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/frames/index_all.csv", help="CSV index of frames")
    parser.add_argument("--out_dir", default="outputs", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    args = parser.parse_args()
    main(args)