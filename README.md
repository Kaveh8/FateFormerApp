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

# FateFormer Explorer

**FateFormer** is a multimodal model (RNA expression, chromatin accessibility, metabolic flux) trained to predict single-cell fate during reprogramming. Labels come from **CellTag-Multi** lineage tracing on a MEF → induced endoderm progenitor (iEP) system.

This repository is the **Streamlit app** that explores the model and data: validation latent space (UMAP), global feature importance (latent shift and attention), per-cell views, and flux-focused analysis. The UI reads precomputed artifacts under `streamlit_hf/cache/`.

**Live app:** [https://huggingface.co/spaces/Angione-Lab/FateFormerExplorer](https://huggingface.co/spaces/Angione-Lab/FateFormerExplorer)

**Run, Docker, Hugging Face Spaces, and cache regeneration:** see [`streamlit_hf/README.md`](streamlit_hf/README.md) and [`streamlit_hf/HUGGINGFACE.md`](streamlit_hf/HUGGINGFACE.md).
