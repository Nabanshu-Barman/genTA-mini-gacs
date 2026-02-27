#!/usr/bin/env python3
"""
Combine all data/frames/*/metadata.csv into data/frames/index_all.csv
Usage:
python src\combine_index.py
"""
import pandas as pd
from pathlib import Path

def combine(frames_root="data/frames", out_csv="data/frames/index_all.csv"):
    root = Path(frames_root)
    all_dfs = []
    for meta in root.rglob("metadata.csv"):
        try:
            df = pd.read_csv(meta)
            all_dfs.append(df)
        except Exception as e:
            print(f"Skipping {meta}: {e}")
    if not all_dfs:
        print("No metadata files found.")
        return
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(out_csv, index=False)
    print(f"Wrote combined index to {out_csv} with {len(combined)} rows")

if __name__ == "__main__":
    combine()