#!/usr/bin/env python3
"""
One-off cache builder for the Streamlit explorer.
Run from the repository root:
  python scripts/precompute_streamlit_cache.py
  python scripts/precompute_streamlit_cache.py --skip-attention   # faster: reuse objects/fi_shift_*.pkl only for df_features if attention_summary exists
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import umap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from data import create_dataset  # noqa: E402
from interpretation import attentions as att  # noqa: E402
from interpretation import latentspace as ls  # noqa: E402
from interpretation import predictions as prds  # noqa: E402
CACHE = ROOT / "streamlit_hf" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def replace_fold_results_path(fold_results, ckp_root: str = "ckp"):
    """Point checkpoints at flat `ckp/multi_seed0_fold{k}.pth` layout in this repo."""
    for fold in fold_results:
        ckpt_name = os.path.basename(fold["best_model_path"])
        fold_token = next((part for part in ckpt_name.split("_") if part.startswith("fold")), "")
        fold_idx = "".join(ch for ch in fold_token if ch.isdigit())
        if fold_idx:
            clean_ckpt_name = f"multi_seed0_fold{fold_idx}.pth"
        else:
            clean_ckpt_name = ckpt_name
        fold["best_model_path"] = os.path.join(ckp_root, clean_ckpt_name)
    return fold_results


def load_training_context():
    with open(ROOT / "objects" / "mutlimodal_dataset.pkl", "rb") as f:
        md = pickle.load(f)
    X, y_label = md["X"], md["y_label"]
    b, df_indices, pcts = md["b"], md["df_indices"], md["pcts"]

    y_number = torch.tensor(
        [{"reprogramming": 1, "dead-end": 0}[i] for i in list(y_label)],
        dtype=torch.float32,
    )
    multimodal_dataset = create_dataset.MultiModalDataset(
        X, b, y_number, df_indices, pcts, y_label
    )

    with open(ROOT / "objects" / "fold_results_multi.pkl", "rb") as f:
        fold_results = pickle.load(f)
    fold_results = replace_fold_results_path(fold_results)

    share_config = {
        "d_model": 128,
        "d_ff": 16,
        "n_heads": 8,
        "n_encoder_layers": 2,
        "n_batches": 3,
        "dropout_rate": 0.0,
    }
    model_config_rna = {"vocab_size": 5914, "seq_len": X[0].shape[1]}
    model_config_atac = {"vocab_size": 1, "seq_len": X[1].shape[1]}
    model_config_flux = {"vocab_size": 1, "seq_len": X[2].shape[1]}
    model_config_multi = {"d_model": 128, "n_heads_cls": 8, "d_ff_cls": 16}
    model_config = {
        "Share": share_config,
        "RNA": model_config_rna,
        "ATAC": model_config_atac,
        "Flux": model_config_flux,
        "Multi": model_config_multi,
    }

    feature_names = (
        list(X[0].columns)
        + ["batch_rna"]
        + list(X[1].columns)
        + ["batch_atac"]
        + list(X[2].columns)
        + ["batch_flux"]
    )

    adata_RNA_labelled = None
    rna_pkl = ROOT / "data" / "datasets" / "rna_labelled.pkl"
    try:
        with open(rna_pkl, "rb") as f:
            adata_RNA_labelled = pickle.load(f)
    except Exception as e:
        print(
            f"Warning: could not load {rna_pkl} ({e}). "
            "Sample table will omit AnnData-derived metadata (e.g. clone_id)."
        )

    return (
        multimodal_dataset,
        fold_results,
        model_config,
        feature_names,
        adata_RNA_labelled,
    )


def build_latent_umap(multimodal_dataset, fold_results, model_config, common_samples: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ls_v, labels, preds = ls.get_latent_space(
        "Multi",
        fold_results,
        multimodal_dataset,
        model_config,
        device,
        common_samples=common_samples,
    )
    reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=30, min_dist=1.0)
    xy = reducer.fit_transform(ls_v)

    ordered_indices: list[int] = []
    fold_ids: list[int] = []
    from interpretation.attentions import filter_idx  # noqa: PLC0415
    from torch.utils.data import Subset  # noqa: PLC0415

    for fold_idx, fold in enumerate(fold_results):
        val_idx = fold["val_idx"]
        if common_samples:
            val_idx = filter_idx(multimodal_dataset, val_idx)
        ordered_indices.extend(val_idx)
        fold_ids.extend([fold_idx + 1] * len(val_idx))

    labels = np.asarray(labels).ravel()
    preds = np.asarray(preds).ravel().astype(int)
    label_name = np.where(labels > 0.5, "reprogramming", "dead-end")
    pred_name = np.where(preds > 0.5, "reprogramming", "dead-end")
    correct = (preds == labels.astype(int)).astype(np.int8)

    ds = multimodal_dataset
    batch_no = np.array([int(ds.batch_no[i].item()) for i in ordered_indices], dtype=np.int32)
    pcts = np.array([float(ds.pcts[i]) for i in ordered_indices], dtype=np.float64)

    modalities = []
    for i in ordered_indices:
        has_r = (ds.rna_data[i] != 0).any().item()
        has_a = (ds.atac_data[i] != 0).any().item()
        has_f = (ds.flux_data[i] != 0).any().item()
        s = "".join(c for c, h in (("R", has_r), ("A", has_a), ("F", has_f)) if h)
        modalities.append(s or "None")

    return {
        "umap_x": xy[:, 0].astype(np.float32),
        "umap_y": xy[:, 1].astype(np.float32),
        "label_name": label_name,
        "pred_name": pred_name,
        "correct": correct,
        "fold": np.array(fold_ids, dtype=np.int32),
        "batch_no": batch_no,
        "pct": pcts,
        "modality": modalities,
        "dataset_idx": np.array(ordered_indices, dtype=np.int32),
        "common_samples": common_samples,
    }


def create_combined_feature_dataframe(
    fi_shift_rna,
    fi_shift_atac,
    fi_shift_flux,
    fi_att_rna,
    fi_att_atac,
    fi_att_flux,
    df_rna_degs=None,
    df_atac_degs=None,
    df_flux_degs=None,
    remove_batch=True,
):
    def process_modality(shift_list, att_list, degs_df, modality_name):
        shift_df = pd.DataFrame(shift_list, columns=["feature", "importance_shift"]).reset_index()
        shift_df.rename(columns={"index": "rank_shift_in_modal"}, inplace=True)
        shift_df["rank_shift_in_modal"] += 1

        att_df = pd.DataFrame(att_list, columns=["feature", "importance_att"]).reset_index()
        att_df.rename(columns={"index": "rank_att_in_modal"}, inplace=True)
        att_df["rank_att_in_modal"] += 1

        combined_df = pd.merge(shift_df, att_df, on="feature", how="outer")
        if degs_df is not None:
            combined_df = pd.merge(combined_df, degs_df, on="feature", how="left")
        combined_df["modality"] = modality_name
        return combined_df

    rna_df = process_modality(fi_shift_rna, fi_att_rna, df_rna_degs, "RNA")
    atac_df = process_modality(fi_shift_atac, fi_att_atac, df_atac_degs, "ATAC")
    flux_df = process_modality(fi_shift_flux, fi_att_flux, df_flux_degs, "Flux")
    all_features_df = pd.concat([rna_df, atac_df, flux_df], ignore_index=True)

    if remove_batch:
        all_features_df = all_features_df[~all_features_df["feature"].str.contains("batch", na=False)]

    max_rank_modal = max(
        all_features_df["rank_att_in_modal"].max(), all_features_df["rank_shift_in_modal"].max()
    )
    all_features_df[["rank_att_in_modal", "rank_shift_in_modal"]] = all_features_df[
        ["rank_att_in_modal", "rank_shift_in_modal"]
    ].fillna(max_rank_modal + 1)
    all_features_df[["rank_att_in_modal", "rank_shift_in_modal"]] = all_features_df[
        ["rank_att_in_modal", "rank_shift_in_modal"]
    ].astype("int32")

    all_features_df[["importance_att", "importance_shift"]] = (
        all_features_df[["importance_att", "importance_shift"]].fillna(0).astype("float64")
    )

    all_features_df["rank_shift"] = (
        all_features_df["importance_shift"].rank(ascending=False, method="first").astype("int32")
    )
    all_features_df["rank_att"] = (
        all_features_df["importance_att"].rank(ascending=False, method="first").astype("int32")
    )
    all_features_df["mean_rank"] = all_features_df[["rank_att", "rank_shift"]].mean(axis=1)

    top_th = int(all_features_df.shape[0] * 0.1) + 1
    all_features_df["top_10_pct"] = all_features_df.apply(
        lambda row: "both"
        if row["rank_shift"] <= top_th and row["rank_att"] <= top_th
        else (
            "shift"
            if row["rank_shift"] <= top_th
            else ("att" if row["rank_att"] <= top_th else "None")
        ),
        axis=1,
    )

    float_cols = [
        col for col in all_features_df.columns if col.startswith(("log_fc", "mean_", "std_", "pval_"))
    ]
    if float_cols:
        all_features_df[float_cols] = all_features_df[float_cols].round(6)
    all_features_df["importance_att"] = all_features_df["importance_att"].round(6)
    all_features_df["importance_shift"] = all_features_df["importance_shift"].round(6)
    all_features_df = all_features_df.sort_values(by="mean_rank", ascending=True)

    cols = [
        "mean_rank",
        "feature",
        "rank_shift",
        "rank_att",
        "rank_shift_in_modal",
        "rank_att_in_modal",
        "modality",
        "importance_shift",
        "importance_att",
        "top_10_pct",
        "mean_de",
        "mean_re",
        "std_de",
        "std_re",
        "pval",
        "pval_adj",
        "log_fc",
        "group",
        "pval_adj_log",
        "mean_diff",
        "pathway",
        "module",
    ]
    for c in cols:
        if c not in all_features_df.columns:
            all_features_df[c] = np.nan
    return all_features_df[cols]


def run_attention_and_fi(
    multimodal_dataset,
    fold_results,
    model_config,
    feature_names,
    device: str,
    adata_rna,
):
    df_samples = prds.get_sample_predictions_dataframe(
        model_type="Multi",
        multimodal_dataset=multimodal_dataset,
        fold_results=fold_results,
        model_config=model_config,
        device=device,
        batch_size=32,
        threshold=0.5,
        adata_rna=adata_rna,
    )
    all_indices = df_samples["ind"].tolist()
    de_preds_indices = df_samples[df_samples["predicted_class"] == "dead-end"]["ind"].tolist()
    re_preds_indices = df_samples[df_samples["predicted_class"] == "reprogramming"]["ind"].tolist()

    print("Running flow attention (all validation)…")
    all_layers_all = att.analyze_cls_attention(
        "Multi",
        fold_results,
        multimodal_dataset,
        model_config,
        device=device,
        indices=all_indices,
        average_heads=False,
        return_flow_attention=True,
    )
    print("Running flow attention (predicted dead-end)…")
    all_layers_de = att.analyze_cls_attention(
        "Multi",
        fold_results,
        multimodal_dataset,
        model_config,
        device=device,
        indices=de_preds_indices,
        average_heads=False,
        return_flow_attention=True,
    )
    print("Running flow attention (predicted reprogramming)…")
    all_layers_re = att.analyze_cls_attention(
        "Multi",
        fold_results,
        multimodal_dataset,
        model_config,
        device=device,
        indices=re_preds_indices,
        average_heads=False,
        return_flow_attention=True,
    )

    rollout_all = att.multimodal_attention_rollout(all_layers_all)
    rollout_de = att.multimodal_attention_rollout(all_layers_de)
    rollout_re = att.multimodal_attention_rollout(all_layers_re)
    rollout_all = rollout_all / rollout_all.sum(dim=-1, keepdim=True)
    rollout_de = rollout_de / rollout_de.sum(dim=-1, keepdim=True)
    rollout_re = rollout_re / rollout_re.sum(dim=-1, keepdim=True)

    # Explicit splits (notebook): RNA [:945], ATAC [945:945+884], flux rest
    i0, i1, i2 = 0, 945, 945 + 884

    def mean_vec(t):
        return t.mean(dim=0).detach().cpu().numpy()

    rollout_mean = {
        "all": mean_vec(rollout_all),
        "dead_end": mean_vec(rollout_de),
        "reprogramming": mean_vec(rollout_re),
    }

    top_n_get = None
    fi = {"all": {}, "dead_end": {}, "reprogramming": {}}
    for name, tensor in (
        ("all", rollout_all),
        ("dead_end", rollout_de),
        ("reprogramming", rollout_re),
    ):
        fi[name]["rna"] = att.get_top_features(
            tensor[:, i0:i1], feature_names[i0:i1], modality="RNA", top_n=top_n_get
        )
        fi[name]["atac"] = att.get_top_features(
            tensor[:, i1:i2], feature_names[i1:i2], modality="ATAC", top_n=top_n_get
        )
        fi[name]["flux"] = att.get_top_features(
            tensor[:, i2:], feature_names[i2:], modality="Flux", top_n=top_n_get
        )

    summary = {
        "feature_names": feature_names,
        "slices": {
            "RNA": {"start": i0, "stop": i1},
            "ATAC": {"start": i1, "stop": i2},
            "Flux": {"start": i2, "stop": len(feature_names)},
        },
        "rollout_mean": rollout_mean,
        "fi_att": fi,
    }
    return summary, df_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-attention", action="store_true", help="Skip attention if summary exists")
    ap.add_argument(
        "--common-samples",
        action="store_true",
        help="Use common-samples filter for latent UMAP (default: False, notebook-style)",
    )
    args = ap.parse_args()
    common_samples = args.common_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    (
        multimodal_dataset,
        fold_results,
        model_config,
        feature_names,
        adata_RNA_labelled,
    ) = load_training_context()

    print("Building latent UMAP bundle…")
    latent = build_latent_umap(
        multimodal_dataset, fold_results, model_config, common_samples=common_samples
    )
    with open(CACHE / "latent_umap.pkl", "wb") as f:
        pickle.dump(latent, f)

    att_path = CACHE / "attention_summary.pkl"
    df_samples_path = CACHE / "samples.parquet"

    if args.skip_attention and att_path.is_file():
        print("Skipping attention (--skip-attention, file exists).")
        with open(att_path, "rb") as f:
            summary = pickle.load(f)
    else:
        print("Computing attention + rollout (slow)…")
        summary, df_samples = run_attention_and_fi(
            multimodal_dataset,
            fold_results,
            model_config,
            feature_names,
            device,
            adata_RNA_labelled,
        )
        with open(att_path, "wb") as f:
            pickle.dump(summary, f)
        with open(CACHE / "attention_feature_ranks.pkl", "wb") as f:
            pickle.dump(summary["fi_att"], f)
        df_samples.to_parquet(df_samples_path, index=False)

    if args.skip_attention and att_path.is_file() and not df_samples_path.is_file():
        df_samples = prds.get_sample_predictions_dataframe(
            model_type="Multi",
            multimodal_dataset=multimodal_dataset,
            fold_results=fold_results,
            model_config=model_config,
            device=device,
            batch_size=32,
            threshold=0.5,
            adata_rna=adata_RNA_labelled,
        )
        df_samples.to_parquet(df_samples_path, index=False)

    for name in ["fi_shift_rna.pkl", "fi_shift_atac.pkl", "fi_shift_flux.pkl"]:
        src = ROOT / "objects" / name
        if not src.is_file():
            print(f"Warning: missing {src}")

    with open(ROOT / "objects" / "fi_shift_rna.pkl", "rb") as f:
        fi_shift_rna = pickle.load(f)
    with open(ROOT / "objects" / "fi_shift_atac.pkl", "rb") as f:
        fi_shift_atac = pickle.load(f)
    with open(ROOT / "objects" / "fi_shift_flux.pkl", "rb") as f:
        fi_shift_flux = pickle.load(f)

    with open(ROOT / "objects" / "degs.pkl", "rb") as f:
        degs = pickle.load(f)
    df_rna_degs, df_atac_degs, df_flux_degs = degs[0], degs[1], degs[2]

    fi = summary["fi_att"]
    df_features = create_combined_feature_dataframe(
        fi_shift_rna,
        fi_shift_atac,
        fi_shift_flux,
        fi["all"]["rna"],
        fi["all"]["atac"],
        fi["all"]["flux"],
        df_rna_degs,
        df_atac_degs,
        df_flux_degs,
    )
    df_features.to_parquet(CACHE / "df_features.parquet", index=False)
    df_features.to_csv(ROOT / "analysis" / "df_features.csv", index=False)
    print(f"Wrote {CACHE / 'df_features.parquet'} and analysis/df_features.csv")
    print("Done.")


if __name__ == "__main__":
    main()
