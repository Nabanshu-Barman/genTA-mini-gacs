#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

def check_metadata(path):
    df = pd.read_csv(path)
    print(f"{path}: rows={len(df)}")
    if len(df) == 0:
        print("WARNING: metadata empty")
    # check files exist
    missing = [p for p in df['file_path'] if not Path(p).exists()]
    if missing:
        print(f"ERROR: {len(missing)} missing files (showing up to 5): {missing[:5]}")
    else:
        print("All files listed in metadata exist.")
    print("Sample rows:")
    print(df.head(3).to_string(index=False))
    if {'width','height','aspect_ratio'}.issubset(df.columns):
        print(df[['width','height','aspect_ratio']].describe())
    else:
        print("Width/height/aspect_ratio columns not found in metadata.")

if __name__ == '__main__':
    base = Path("data/frames")
    any_meta = False
    for d in base.iterdir():
        meta = d / "metadata.csv"
        if meta.exists():
            any_meta = True
            check_metadata(meta)
    combined = base / "index_all.csv"
    if combined.exists():
        print("\nCombined index summary:")
        check_metadata(combined)
    else:
        if not any_meta:
            print("No per-video metadata files found. Run extraction first.")
        else:
            print("Combined index not found. Run src/combine_index.py")