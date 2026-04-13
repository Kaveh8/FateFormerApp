---
title: FateFormer Explorer
short_description: Streamlit app to explore multimodal single-cell fate modeling (RNA, ATAC, metabolic flux, attention, and rankings).
emoji: 🧬
colorFrom: violet
colorTo: indigo
tags:
  - streamlit
  - single-cell
  - multi-omics
  - genomics
  - atac-seq
  - rna-seq
  - metabolic-modeling
  - deep-learning
  - biology
license: mit
sdk: docker
app_port: 7860
---

# FateFormerApp

## Interactive explorer (Streamlit)

From the repo root, with the project virtualenv activated:

```bash
PYTHONPATH=. streamlit run streamlit_hf/app.py
```

The default local port is **8501**. The **Dockerfile** (and Hugging Face Space card above) use **7860** to match Spaces.

### Updating results after new experiments (no code changes)

The app reads **fixed paths**. Replace files under `streamlit_hf/cache/` using the **same filenames**; then **restart Streamlit** (or do a hard refresh) so the new data loads.

| File | What it drives |
|------|----------------|
| `streamlit_hf/cache/latent_umap.pkl` | Single-Cell Explorer (UMAP) |
| `streamlit_hf/cache/df_features.parquet` | Feature insights + Flux analysis |
| `streamlit_hf/cache/attention_summary.pkl` | “Attention vs prediction” in Feature insights |
| `streamlit_hf/cache/attention_feature_ranks.pkl` | Optional; attention lists also live inside `attention_summary.pkl` |

You can also keep `analysis/df_features.csv` in sync for your own workflows; the UI **prefers** `streamlit_hf/cache/df_features.parquet` when present.

### Regenerating caches from this repo

If you updated checkpoints, fold splits, shift pickles, or deg tables **inside this project**, run:

```bash
python scripts/precompute_streamlit_cache.py
```

That script expects (among others) `ckp/*.pth`, `objects/fold_results_multi.pkl`, `objects/mutlimodal_dataset.pkl`, `objects/fi_shift_*.pkl`, and `objects/degs.pkl`. Point those inputs at your new experiment outputs **before** running the script, or copy your new pickles/CSVs into `streamlit_hf/cache/` manually as in the table above.

### Docker / Hugging Face

See `streamlit_hf/HUGGINGFACE.md` and the root `Dockerfile`.
