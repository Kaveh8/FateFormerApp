"""Load precomputed explorer artifacts (no torch required at runtime)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from streamlit_hf.lib.formatters import annotate_modality_column
from streamlit_hf.lib.reactions import normalize_reaction_key

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "streamlit_hf" / "cache"
METABOLIC_MODEL_METADATA = REPO_ROOT / "data" / "datasets" / "metabolic_model_metadata.csv"


def _is_valid_features_csv(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        head = pd.read_csv(path, nrows=2)
    except Exception:
        return False
    return "feature" in head.columns and "importance_shift" in head.columns


def load_latent_bundle():
    path = CACHE_DIR / "latent_umap.pkl"
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_attention_summary():
    path = CACHE_DIR / "attention_summary.pkl"
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_samples_df() -> pd.DataFrame | None:
    pq = CACHE_DIR / "samples.parquet"
    if pq.is_file():
        df = pd.read_parquet(pq)
        return annotate_modality_column(df) if "modality" in df.columns else df
    return None


def _add_within_modality_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align scatter / table columns with the notebook.

    Parquet from precompute already has rank_shift_in_modal / rank_att_in_modal from the same
    merge-of-sorted-lists logic as the notebook; do not overwrite those with pandas ranks on
    rounded importances (tie order can differ and changes the RNA cloud).
    """
    out = df.copy()
    if "modality" not in out.columns:
        return out
    if "rank_shift_in_modal" in out.columns and "rank_att_in_modal" in out.columns:
        out["shift_order_mod"] = out["rank_shift_in_modal"].astype(int)
        out["attention_order_mod"] = out["rank_att_in_modal"].astype(int)
    else:
        g = out.groupby("modality", observed=True)
        out["shift_order_mod"] = g["importance_shift"].rank(ascending=False, method="first").astype(int)
        out["attention_order_mod"] = g["importance_att"].rank(ascending=False, method="first").astype(int)
        out["rank_shift_in_modal"] = out["shift_order_mod"]
        out["rank_att_in_modal"] = out["attention_order_mod"]
    if "combined_order_mod" not in out.columns:
        g = out.groupby("modality", observed=True)
        out["combined_order_mod"] = g["mean_rank"].rank(ascending=True, method="first").astype(int)
    return out


def load_metabolic_model_metadata() -> pd.DataFrame | None:
    """Directed reaction edges: substrate → product, grouped by supermodule (see CSV headers)."""
    if not METABOLIC_MODEL_METADATA.is_file():
        return None
    return pd.read_csv(METABOLIC_MODEL_METADATA)


def build_metabolic_model_table(
    meta: pd.DataFrame,
    flux_df: pd.DataFrame,
    supermodule_id: int | None = None,
) -> pd.DataFrame:
    """
    Static edge list: substrate → product, reaction label, module class, plus DE / model columns when the
    reaction string matches a row in the flux feature table.
    """
    need = {"Compound_IN_name", "Compound_OUT_name", "rxnName", "Supermodule_id", "Super.Module.class"}
    if not need.issubset(set(meta.columns)):
        return pd.DataFrame()
    m = meta.copy()
    if supermodule_id is not None:
        m = m[m["Supermodule_id"] == int(supermodule_id)]
    if m.empty:
        return pd.DataFrame()

    fd = flux_df.copy()
    fd["_rk"] = fd["feature"].map(normalize_reaction_key)
    fd = fd.drop_duplicates("_rk", keep="first").set_index("_rk", drop=False)

    rows: list[dict] = []
    for _, r in m.iterrows():
        k = normalize_reaction_key(str(r["rxnName"]))
        base = {
            "Supermodule": r.get("Super.Module.class"),
            "Module_id": r.get("Module_id"),
            "Substrate": r["Compound_IN_name"],
            "Product": r["Compound_OUT_name"],
            "Reaction": r["rxnName"],
        }
        if k in fd.index:
            row = fd.loc[k]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            base["log_fc"] = row["log_fc"] if "log_fc" in row.index else None
            base["pval_adj"] = row["pval_adj"] if "pval_adj" in row.index else None
            base["mean_rank"] = row["mean_rank"] if "mean_rank" in row.index else None
            base["pathway"] = row["pathway"] if "pathway" in row.index else None
        else:
            base["log_fc"] = None
            base["pval_adj"] = None
            base["mean_rank"] = None
            base["pathway"] = None
        rows.append(base)
    return pd.DataFrame(rows)


def load_df_features() -> pd.DataFrame | None:
    pq = CACHE_DIR / "df_features.parquet"
    if pq.is_file():
        return _add_within_modality_orders(pd.read_parquet(pq))
    csv_cache = CACHE_DIR / "df_features.csv"
    if csv_cache.is_file():
        return _add_within_modality_orders(pd.read_csv(csv_cache))
    analysis_csv = REPO_ROOT / "analysis" / "df_features.csv"
    if _is_valid_features_csv(analysis_csv):
        return _add_within_modality_orders(pd.read_csv(analysis_csv))
    return None


def latent_join_samples(bundle: dict, samples: pd.DataFrame | None) -> pd.DataFrame:
    """One row per UMAP point, aligned with bundle arrays."""
    n = len(bundle["umap_x"])
    df = pd.DataFrame(
        {
            "umap_x": bundle["umap_x"],
            "umap_y": bundle["umap_y"],
            "label": bundle["label_name"],
            "predicted_class": bundle["pred_name"],
            "correct": bundle["correct"].astype(bool),
            "fold": bundle["fold"].astype(int),
            "batch_no": bundle["batch_no"].astype(int),
            "pct": bundle["pct"],
            "modality": bundle["modality"],
            "dataset_idx": bundle["dataset_idx"].astype(int),
        }
    )
    if samples is not None and not samples.empty:
        s = samples.drop_duplicates(subset=["ind"], keep="first").set_index("ind")
        extra = s.reindex(df["dataset_idx"].values)
        for col in ["predicted_value", "clone_id", "clone_size", "cell_type"]:
            if col in extra.columns:
                df[col] = extra[col].values
    return annotate_modality_column(df)
