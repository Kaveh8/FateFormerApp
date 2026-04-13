# Hugging Face Space (Docker + Streamlit)

The **root `README.md`** starts with the YAML card Hugging Face reads for the Space (title, tags, colours, `sdk: docker`, `app_port: 7860`). Copy that block if you maintain a separate Space README.

```yaml
---
title: FateFormer Explorer
short_description: Multimodal fate from RNA, ATAC, and metabolic flux models.
emoji: 🧬
colorFrom: purple
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
```

`app_port` **7860** matches the root **`Dockerfile`** (`streamlit ... --server.port 7860`). Local runs use Streamlit’s default **8501** unless you pass `--server.port`.

## Before first deploy

1. Run locally: `python scripts/precompute_streamlit_cache.py` (requires GPU/CPU time for attention).
2. Commit **`streamlit_hf/cache/`** contents (`latent_umap.pkl`, `attention_summary.pkl`, `attention_feature_ranks.pkl`, `df_features.parquet`, and optionally `samples.parquet` if you use it elsewhere) or attach via **Git LFS** if files are large. These paths are listed in `.gitignore`; use `git add -f streamlit_hf/cache/*` when you want them in the remote.
3. Keep **`ckp/`** model weights available only if you run precompute in CI; the slim Docker image does **not** include PyTorch and expects precomputed caches.

The repository **`Dockerfile`** at the root builds the Space.
