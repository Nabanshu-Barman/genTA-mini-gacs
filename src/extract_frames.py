#!/usr/bin/env python3
"""
Extract frames from a video at a regular interval and write metadata CSV.

Usage example:
python src\extract_frames.py --input "data/raw/video1.mp4" --out_dir "data/frames/video1" --video_id v1 --interval 1.5 --max_frames 25
"""
import argparse
from pathlib import Path
import cv2
import csv
import hashlib
from PIL import Image
from tqdm import tqdm

def compute_md5(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_frames_interval(video_path: Path, out_dir: Path, interval_sec: float, video_id: str, max_frames: int = None):
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if fps > 0 else 0.0

    # sample timestamps at 0, interval, 2*interval, ... until duration
    timestamps = []
    t = 0.0
    while t <= max(duration_sec - 1e-3, 0.0):
        timestamps.append(t)
        t += interval_sec

    # If max_frames specified and timestamps exceed it, uniformly downsample timestamps
    if max_frames is not None and len(timestamps) > max_frames:
        import numpy as np
        idxs = np.linspace(0, len(timestamps) - 1, num=max_frames, dtype=int)
        timestamps = [timestamps[i] for i in idxs]

    meta_rows = []
    pbar = tqdm(timestamps, desc=f"Extracting frames ({video_id})", unit="frame")
    for ts in pbar:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts * 1000.0))
        success, frame = cap.read()
        if not success or frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        width, height = img.size
        aspect_ratio = width / height if height != 0 else 0.0
        timestamp_ms = int(round(ts * 1000))
        fname = f"{video_id}_{timestamp_ms:07d}.jpg"
        out_path = out_dir / fname
        img.save(out_path, quality=95)
        md5 = compute_md5(out_path)
        meta_rows.append({
            "video_id": video_id,
            "source_path": str(video_path),
            "timestamp_s": f"{ts:.3f}",
            "file_path": str(out_path),
            "width": width,
            "height": height,
            "aspect_ratio": f"{aspect_ratio:.6f}",
            "md5": md5
        })

    cap.release()
    meta_csv = out_dir / "metadata.csv"
    if meta_rows:
        with open(meta_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
            writer.writeheader()
            writer.writerows(meta_rows)
    else:
        with open(meta_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id","source_path","timestamp_s","file_path","width","height","aspect_ratio","md5"])
    print(f"Extracted {len(meta_rows)} frames to {out_dir} and wrote metadata {meta_csv}")
    return meta_rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--out_dir", required=True, help="Directory to write frames and metadata.csv")
    parser.add_argument("--video_id", required=True, help="Short id for video (used in filenames)")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between sampled frames")
    parser.add_argument("--max_frames", type=int, default=None, help="If set, limit the number of output frames (uniform downsample)")
    args = parser.parse_args()

    extract_frames_interval(
        video_path=Path(args.input),
        out_dir=Path(args.out_dir),
        interval_sec=args.interval,
        video_id=args.video_id,
        max_frames=args.max_frames
    )