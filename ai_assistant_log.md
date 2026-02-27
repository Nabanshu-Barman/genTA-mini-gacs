# AI Assistant Governance — Mini GACS Prototype (GenTA Competency Assessment)

**Date (file created):** 2026-02-27  
**Primary Engineer / Author:** Nabanshu Barman (GitHub: Nabanshu-Barman)  
**AI Tools referenced in this log:**  
- GitHub Copilot (primary coding assistant)  
- ChatGPT (secondary, conceptual checks)

---

## Purpose

This single-file governance & provenance log documents where AI assistants were used during development of the Mini GACS Prototype (Mood & Style Embedding Pipeline). It records:

- Which AI tool was used and when  
- Example (mock) prompts that were used or could reasonably have been used  
- The raw responses (representative / mock) returned by the assistant  
- What I changed after accepting/guiding the suggestions  
- Why I made those changes

The goal is a transparent, reproducible audit trail showing that AI was used as an assistant and that final decisions, verification, and validation were performed by me.

---

## Executive Summary

I designed and implemented an end-to-end pipeline to:

1. Ingest 2–3 short royalty-free art/marketing-style videos  
2. Extract representative frames (interval sampling)  
3. Compute fixed-size embeddings using a pre-trained vision model (CLIP)  
4. Compute pairwise cosine similarity across frames  
5. Retrieve top-k similar frames for queries  
6. Visualize similarities (heatmap, t-SNE) and show query + neighbors

Design decisions, verification strategy (assertions, unit tests, visual inspection), and the repository structure were created and executed by me. GitHub Copilot helped accelerate repetitive and boilerplate coding; ChatGPT was used occasionally for conceptual clarifications. All assistant outputs were audited, adapted, and verified before being accepted.

---

## Repository snapshot (relevant files)

- data/
  - raw/
  - frames/
- notebooks/
  - demo_pipeline.ipynb
- outputs/
  - embeddings.npy
  - embeddings_index.csv
  - similarity.npy
  - heatmap.png
  - tsne.png
  - topk.csv
- report/
  - demo_pipeline.html
  - GenTA_mini_gacs_report.pdf
- src/
  - extract_frames.py
  - compute_embeddings.py
  - similarity.py
  - inspect_results.py
- tests/
  - test_embeddings.py
- ai_assistant_log.md (this file)
- README.md
- DATA_LICENSES.md

---

## Phase-wise Summary (high level)

- Phase 0 — Planning & design (human-led): stack, outputs, verification-first goals.
- Phase 1 — Frame extraction (implemented; Copilot assisted for CLI and loops).
- Phase 2 — Embedding computation using CLIP (implemented; Copilot suggested boilerplate; I added batching, normalization, asserts).
- Phase 3 — Similarity, top-k retrieval, visualizations (implemented; Copilot suggested plotting snippets; I added sanity checks and manual verification).
- Tests: Small pytest tests for embedding sanity and duplicate-image checks.
- Deliverables: notebook, small dataset or download script, outputs (heatmap, tsne), short screen recording and written report.

---

## AI Governance & Detailed Interaction Log

Below are representative (mock) interaction entries showing how AI tools were used during development. Each entry follows the form:

- Date  
- Tool  
- Prompt (mock / representative)  
- Raw response (representative excerpt)  
- What I modified  
- Why I modified it

Notes:
- These prompts/responses are representative reconstructions used for governance and reproducibility. They reflect the kinds of prompts I used when interacting with Copilot and ChatGPT during development.
- Tools used: only GitHub Copilot and ChatGPT.

---

### Interaction 1
Date: 2026-02-20  
Tool: GitHub Copilot  
Prompt: "Create a Python CLI skeleton for extract_frames.py that takes --input_dir, --output_dir, --interval and iterates videos saving one frame every N seconds."  

Raw response (representative excerpt):
```python
import argparse
import os
from pathlib import Path
import cv2

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--interval', type=float, default=2.0)
    return p.parse_args()

def extract_frames_from_video(video_path, out_dir, interval):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            out_path = out_dir / f"{video_path.stem}_{idx}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved

if __name__ == "__main__":
    args = parse_args()
    for v in Path(args.input_dir).glob("*.mp4"):
        extract_frames_from_video(v, Path(args.output_dir), args.interval)
```

What I modified:
- Replaced naive frame index-based naming with timestamp-based filenames (milliseconds) to make metadata exact.
- Added PIL open validation after save to check file integrity.
- Added MD5 or SHA1 computation per saved frame for duplicate detection.
- Added CSV metadata writing (idx, video_id, timestamp_s, file_path, width, height, md5).
- Replaced raw cv2.imencode usage to ensure consistent JPEG quality settings.

Why:
- Timestamped filenames simplify alignment and deduplication.
- Metadata (timestamps, checksums) are required for reproducibility and verification.
- Adding file-open checks prevents silent failures when cv2.imwrite succeeded but saved corrupted files.

---

### Interaction 2
Date: 2026-02-21  
Tool: GitHub Copilot  
Prompt: "Provide a compute_embeddings.py stub that loads images from a frames directory, uses a HuggingFace CLIP model to compute embeddings in batches, and saves outputs to embeddings.npy and embeddings_index.csv."  

Raw response (representative excerpt):
```python
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from PIL import Image
import os, csv

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def compute_embeddings(image_paths, batch_size=32):
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch = [Image.open(p).convert("RGB") for p in image_paths[i:i+batch_size]]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        emb = outputs.cpu().numpy()
        embeddings.append(emb)
    return np.vstack(embeddings)
```

What I modified:
- Added explicit L2 normalization of embeddings before saving (important for cosine similarity).
- Added dtype enforcement to float32, and a NaN check (assert not np.isnan(embeddings).any()).
- Wrapped the HuggingFace processor call to avoid GPU memory leaks (moved tensors off-device promptly).
- Wrote an `embeddings_index.csv` with `idx, file_path, video_id, timestamp_s`.
- Added a deterministic seeding option for reproducibility.

Why:
- Normalization ensures cosine similarity equals dot product on normalized vectors and improves numerical stability.
- NaN checks, dtype, and seeding are verification-first practices that catch runtime issues early.

---

### Interaction 3
Date: 2026-02-22  
Tool: ChatGPT  
Prompt: "What checks should I run to verify image embeddings are valid and comparable with cosine similarity?"  

Raw response (representative excerpt):
- Check embedding shape matches expected dimensionality (e.g., (N, 512)).
- Check dtype is float32 or float64.
- Ensure no NaNs or Infs: np.isnan(...) and np.isinf(...).
- L2-normalize embeddings so cosine similarity is equivalent to dot product.
- Verify identical images produce identical embeddings (or within numerical tolerance).
- Check embedding norms are close to 1 after normalization.
- For stability, check a small sample of pairwise similarities fall in [-1, 1] and diagonal is ≈ 1.

What I modified:
- Converted this guidance into a concrete `verify_phase1.py` script that runs the checks automatically and fails with clear messages.
- Implemented the identical-image test: embed(img) and embed(img_copy) and assert np.allclose(..., atol=1e-6).

Why:
- Translating conceptual checks into automated assertions ensures reproducibility and that future modifications are guarded by tests.

---

### Interaction 4
Date: 2026-02-22  
Tool: GitHub Copilot  
Prompt: "Give me a concise snippet to compute pairwise cosine similarities from normalized embeddings and save the top-5 neighbors for each query as CSV."  

Raw response (representative excerpt):
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# embeddings: shape (N, D) already normalized.
S = cosine_similarity(embeddings)  # NxN

topk = []
k = 5
for i in range(S.shape[0]):
    idxs = np.argsort(-S[i])[:k+1]  # includes self
    for rank, j in enumerate(idxs[1:], start=1):  # skip self
        topk.append((i, int(j), rank, float(S[i, j])))

# save topk list to CSV (query_idx, neighbor_idx, rank, score)
```

What I modified:
- Replaced sklearn cosine_similarity with optimized dot product `S = embeddings @ embeddings.T` because embeddings were normalized; this is faster and avoids dependency variations.
- Added symmetry and diagonal checks: `assert np.allclose(S, S.T, atol=1e-6)` and `assert np.allclose(np.diag(S), 1.0, atol=1e-5)`.
- Clipped numerical values to [-1, 1] for safety.
- Implemented an efficient partial top-k using `np.argpartition` for large N to avoid full sorting when N grows.

Why:
- Dot-product is mathematically equivalent for normalized vectors and faster.
- Symmetry/diagonal checks and clipping prevent weird downstream plotting artifacts and enforce invariants.

---

### Interaction 5
Date: 2026-02-23  
Tool: ChatGPT  
Prompt: "What visualization should I produce to convincingly show 'vibe' similarity across frames? Any pitfalls for t-SNE?"  

Raw response (representative excerpt):
- Use a similarity heatmap (ordered by cluster or original video) to show pairwise relationships.
- Use t-SNE or UMAP for 2D projection of embeddings and color points by video_id or another metadata attribute.
- For t-SNE pitfalls:
  - t-SNE is stochastic; run with a fixed random_state for reproducibility.
  - Poor for extremely large datasets; subsample if necessary.
  - Perplexity and learning_rate parameters can change cluster separation. Try a few values (e.g., perplexity 30).
  - Prefer UMAP for preserving global structure sometimes.
- Label a few query images with top neighbors shown inline to provide a human-perceivable validation of "vibe".

What I modified:
- Implemented both t-SNE and UMAP options in `src/similarity.py`, defaulting to t-SNE with `random_state` set.
- Added a reproducibility note in the notebook and code.
- Subsampled when N > 2000 to keep plotting meaningful and consistent.

Why:
- Following guidance improves interpretability and ensures visual artifacts do not mislead reviewers.
- Deterministic runs are necessary for reproducibility during assessment.

---

### Interaction 6
Date: 2026-02-24  
Tool: GitHub Copilot  
Prompt: "Create a small pytest for embeddings that asserts shape, no-NaN, and identical-image equivalence."  

Raw response (representative excerpt):
```python
def test_embeddings_basic():
    embeddings = np.load("outputs/embeddings.npy")
    assert embeddings.shape[1] == 512
    assert not np.isnan(embeddings).any()
    # identical image test (requires sample image saved as sample.jpg)
    emb1 = embed_image("data/frames/sample.jpg")
    emb2 = embed_image("data/frames/sample_copy.jpg")
    assert np.allclose(emb1, emb2, atol=1e-6)
```

What I modified:
- Made the test robust to model dimension by reading expected dimension from `outputs/embedding_stats.txt`.
- Added setup/teardown logic so tests create small synthetic images if sample images are absent rather than failing.
- Ensured tests do not invoke a GPU-only path on CI (force CPU device in test environment).

Why:
- Tests must be robust and not flaky; adding synthetic-image fallback ensures CI can run without requiring human-provided data.

---

### Interaction 7
Date: 2026-02-25  
Tool: ChatGPT  
Prompt: "How should I write the AI provenance sentence in README for the assessment?"  

Raw response (representative excerpt):
- Example sentence: "GitHub Copilot was used to assist in writing boilerplate code and repetitive tasks; all design decisions, verification checks, and final code approval were performed by [Your Name]."  
- Suggested addition: "A detailed AI assistant log is included in ai_assistant_log.md describing each use, representative prompts, and verification steps."

What I modified:
- Tailored the sentence to be precise: "GitHub Copilot was used as the primary coding assistant for boilerplate and repetitive code; I (Nabanshu Barman) performed all architectural decisions, tests, and final validation. Occasional conceptual checks were performed with ChatGPT. See ai_assistant_log.md for details."

Why:
- Clear, explicit attribution and provenance is required by the assessment instructions and for reviewers to understand the governance.

---

## Consolidated AI Usage Summary

- GitHub Copilot: primary for code scaffolding, CLI boilerplate, loop constructs, plotting templates, and small debugging hints. I reviewed and modified every suggestion before committing.
- ChatGPT: secondary for conceptual verification, best-practices guidance (embedding checks, t-SNE pitfalls), and natural-language suggestions for documentation. I converted conceptual advice into concrete assertions, tests, and deterministic settings.
- All AI-suggested code was inspected, tested, and adapted. No AI output was blindly accepted.

---

## Verification-first checklist (executable assertions used in development)

- File presence:
```python
assert os.path.exists("outputs/embeddings.npy")
assert os.path.exists("outputs/topk.csv")
```

- Embedding sanity:
```python
emb = np.load("outputs/embeddings.npy")
assert emb.dtype == np.float32 or emb.dtype == np.float64
assert not np.isnan(emb).any()
# L2 norms approx 1
norms = np.linalg.norm(emb, axis=1)
assert np.allclose(norms, 1.0, atol=1e-4)
```

- Similarity sanity:
```python
S = emb @ emb.T
assert np.allclose(S, S.T, atol=1e-6)
assert np.all((S >= -1.0) & (S <= 1.0))
assert np.allclose(np.diag(S), 1.0, atol=1e-5)
```

- Identical image embedding equality:
```python
e1 = embed_image("data/frames/x.jpg")
e2 = embed_image("data/frames/x_copy.jpg")
assert np.allclose(e1, e2, atol=1e-6)
```

---

## Reproducibility Notes & Environment

- Python 3.9+
- Requirements listed in `requirements.txt` (torch, torchvision, transformers, numpy, pandas, matplotlib, pillow, scikit-learn, pytest)
- Scripts are CLI-configurable: model_name, batch_size, frame_interval, topk
- Default runs were CPU-friendly for evaluation; GPU optional for speed
- Deterministic seeds set for t-SNE and other stochastic components

---

## Next steps toward a production GACS engine (brief)

1. Expand dataset across creative categories and market verticals.  
2. Add audio and text (OCR) embeddings and implement multimodal fusion.  
3. Collect human-labeled "vibe" scores and use supervised calibration to map embedding-space distances to perceived mood metrics.  
4. Persist embeddings in a production vector store (Faiss/Milvus) and expose API endpoints for real-time retrieval.  
5. Hook up to CTR/ROAS performance signals and build an online feedback loop for continuous improvement.

---

## Closing statement

I, Nabanshu Barman, am the primary author and engineer of this pipeline. GitHub Copilot and ChatGPT were used as assistants to accelerate development and provide conceptual checks. All architectural choices, verification logic, testing, and final acceptance were performed by me. This document provides a transparent record of AI usage, mock prompts, raw responses (representative), and the exact modifications and reasons for them.

---

**End of ai_assistant_log.md**