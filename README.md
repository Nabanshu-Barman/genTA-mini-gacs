# Mini GACS Prototype — Mood & Style Embedding Pipeline  
GenTA Competency Assessment — AI R&D Engineer

**Author:** Nabanshu Barman  
**Date:** 27 February 2026

---

## 1. Overview

This repository contains a fully reproducible, end-to-end prototype of a minimal GACS-like affective computing pipeline tailored to GenTA’s domain: understanding the “feel” of contemporary art and marketing-style visuals.

The system:
- Ingests 2–3 short royalty-free art/marketing videos
- Extracts representative frames
- Computes fixed-size embeddings using a pre-trained vision model (CLIP)
- Computes cosine similarity to estimate mood/style (“vibe”) similarity
- Generates visualizations (heatmap + t-SNE / UMAP)
- Includes unit tests and verification checks
- Provides a demo notebook and a reproducible pipeline

This is not a large-scale model training project. It demonstrates architectural thinking, verification-first engineering, AI governance, and a foundation that could evolve into GenTA’s GACS engine.

---

## 2. How This Aligns With GenTA’s Vision

GenTA’s goal is to make contemporary art and marketing creatives emotionally interpretable and performance-aware.

This prototype demonstrates:
- Style-aware embeddings using a multimodal vision model
- Similarity-based vibe retrieval
- Cluster separation between emotional, stylistic, and high-motion creatives (visualized)
- A verification-first AI workflow with tests and assertions
- A clear extension path toward CTR/ROAS integration

The architecture is modular and production-extensible.

---

## 3. Repository Structure

Project root: `genTA-mini-gacs/`

```
genTA-mini-gacs/
│
├── data/
│   ├── raw/                # Place/download input videos here
│   └── frames/             # Extracted representative frames
│
├── notebooks/
│   └── demo_pipeline.ipynb # Interactive inspection notebook
│
├── outputs/
│   ├── embeddings.npy
│   ├── embeddings_index.csv
│   ├── topk.csv
│   ├── similarity.npy
│   ├── heatmap.png
│   └── tsne.png
│
├── src/
│   ├── extract_frames.py
│   ├── compute_embeddings.py
│   ├── combine_index.py
│   ├── similarity.py
│   └── verify_phase1.py
│
├── tests/
│   ├── test_extract_frames.py
│   └── test_embeddings.py
│
├── report/
│   ├── final_report.pdf
│   └── screen_recording.mp4
│
├── ai_assistant_log.md
├── DATA_LICENSES.md
├── README.md
└── requirements.txt
```

---

## 4. Requirements

- Python 3.9+ (3.10+ recommended)

Core Python packages (see `requirements.txt`):
- torch
- torchvision
- transformers
- numpy
- pandas
- Pillow
- matplotlib
- seaborn
- scikit-learn
- pytest
- tqdm
- imageio-ffmpeg
- umap-learn (optional, recommended for alternative projections)

Install these with:
```bash
pip install -r requirements.txt
```

---

## 5. Quickstart — Run the Full Pipeline

Run the entire pipeline end-to-end.

### Step 1 — Create Environment

Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Add Sample Videos

Place 2–3 short royalty-free videos in:
```
data/raw/
```
Do NOT commit copyrighted videos to the repository. Record license details in `DATA_LICENSES.md`.

### Step 3 — Extract Representative Frames

Example:
```bash
python src/extract_frames.py \
  --input_dir data/raw \
  --output_dir data/frames \
  --interval 2.0
```
What this does:
- Extracts 1 frame every 2 seconds (adjust `--interval`)
- Saves frames to `data/frames/<video_id>/`
- Writes metadata to `outputs/frames_index.csv`
- Aim for ~20–50 frames per video for this prototype

### Step 4 — Compute Embeddings

Example:
```bash
python src/compute_embeddings.py \
  --frames_dir data/frames \
  --out_embeddings outputs/embeddings.npy \
  --out_index outputs/embeddings_index.csv \
  --model_name "openai/clip-vit-base-patch32" \
  --batch_size 32
```
Outputs:
- `embeddings.npy` → shape (N_frames, D) (e.g., D = 512)
- `embeddings_index.csv` (mapping index → metadata)

Sanity checks performed:
- Shapes printed
- No NaNs
- L2 normalization applied
- Duplicate-image embedding equality test

### Step 5 — Compute Similarity & Visualizations

Example:
```bash
python src/similarity.py \
  --embeddings outputs/embeddings.npy \
  --index outputs/embeddings_index.csv \
  --out_topk outputs/topk.csv \
  --out_heatmap outputs/heatmap.png \
  --out_tsne outputs/tsne.png \
  --topk 5
```
Outputs:
- Cosine similarity matrix (saved or computed on-the-fly)
- Top-5 neighbors per query (`outputs/topk.csv`)
- Heatmap visualization (`outputs/heatmap.png`)
- Projection (t-SNE / UMAP) plot (`outputs/tsne.png`)

---

## 6. Reproducibility Steps

To reproduce results:
1. Clone repository
2. Create environment and install requirements
3. Add the same licensed videos to `data/raw/`
4. Run pipeline: extract → embeddings → similarity
5. Run tests: `pytest -q`
6. Open and run the notebook `notebooks/demo_pipeline.ipynb`

Determinism depends on:
- Fixed frame interval
- Fixed random seeds (set in code)
- Same model version (use the same HuggingFace model name)

---

## 7. Run Tests (Verification-First Workflow)

Run:
```bash
pytest -q
```

Included tests:
- `test_no_nans` — ensure no NaNs in embeddings
- `test_embedding_dimensions` — embedding dimensionality matches metadata
- `test_duplicate_images_equal_embeddings` — identical images produce near-identical embeddings
- Similarity symmetry check

These tests enforce verification-first engineering and help catch regressions early.

---

## 8. Demo Notebook Instructions

Open:
```
notebooks/demo_pipeline.ipynb
```
Then:
- Run all cells
- Confirm file existence checks
- View the heatmap and t-SNE plots
- Use the query cells to show a query image + top-5 neighbors
- Inspect CSV outputs (`embeddings_index.csv`, `topk.csv`)

The notebook provides an interactive inspection layer on top of the reproducible scripts.

---

## 9. Expected Runtime

On CPU (no GPU):
- Frame extraction: < 2 minutes (for 2–3 short videos)
- Embedding computation (≈100 frames): ~3–7 minutes
- Similarity + visualization: < 1 minute
- Full pipeline: ~10 minutes (varies by machine)

On GPU:
- Embedding computation time reduces significantly (1–2 minutes for 100 frames)

The repository is designed to complete within ~15 minutes on a typical CPU for the small sample size used here.

---

## 10. Verification-First Engineering Principles Applied

- Fixed random seeds for stochastic algorithms
- Assertions for file existence and shapes
- NaN / Inf detection for embeddings
- L2 normalization of embeddings
- Duplicate-image equality checks
- Similarity matrix invariants (symmetry, diagonal ≈ 1)
- Structured logging and saved metadata (CSV + NumPy arrays)
- Unit tests (pytest) before submission

These practices ensure robustness and auditability.

---

## 11. AI Assistant Usage & Governance

AI tools used:
- GitHub Copilot (primary)
- ChatGPT (conceptual verification)

AI was used for:
- Boilerplate scaffolding (CLIs, loops)
- Batch-processing examples
- Plotting templates
- Minor debugging suggestions

AI was NOT used for:
- Architecture decisions
- Model selection logic
- Verification strategy
- Final interpretation of results

All AI-generated code was reviewed, adapted, and verified by me. See `ai_assistant_log.md` for a detailed audit trail of prompts, representative responses, and modifications.

---

## 12. From Prototype to Production GACS Engine (Next Steps)

1. Multimodal fusion
   - Add audio embeddings (music/voice)
   - Add OCR/text embeddings (on-screen text)
   - Fuse modalities for richer affective signals
2. Performance feedback loop
   - Map embeddings → CTR/ROAS via supervised training
   - Train ranking/regression models to predict performance
3. Real-time serving
   - Persist embeddings in FAISS / Milvus
   - Deploy retrieval API for creative pipelines
4. Human-in-the-loop labeling
   - Collect vibe annotations to refine embedding sensitivity
   - Fine-tune or train contrastive models on vibe-labeled pairs

---

## 13. Deliverables

This repository contains:
- Public GitHub repo (this project)
- 2–4 page written report (`report/final_report.pdf`)
- 2-minute demo video (`report/screen_recording.mp4`)
- Demo notebook (`notebooks/demo_pipeline.ipynb`)
- Outputs folder with embeddings and plots
- AI assistant audit log (`ai_assistant_log.md`)

---

## 14. Troubleshooting

- Out-of-memory / OOM:
  - Reduce `--batch_size` in `compute_embeddings.py`.
  - Run CPU-only if GPU memory is limited.
- Missing files:
  - Ensure `data/frames/` exists and contains images before running embeddings.
- Model download failure:
  - Check internet access.
  - Manually download model from HuggingFace and set `--model_name`.
- Similarity values look incorrect:
  - Verify embeddings are normalized (L2 norm ≈ 1).
  - Check diagonal of similarity matrix is close to 1.

---

## 15. License & Attribution

- Code authored by Nabanshu Barman.
- Pre-trained models used under their respective licenses (HuggingFace / OpenAI CLIP).
- Videos must be sourced under royalty-free or public-domain licenses — record details in `DATA_LICENSES.md`.

---

## 16. Contact

GitHub: [Nabanshu-Barman](https://github.com/Nabanshu-Barman)

For reproducibility questions, issues, or extension requests, please open an issue in this repository.