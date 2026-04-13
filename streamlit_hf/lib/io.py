"""Load precomputed explorer artifacts (no torch required at runtime)."""

from __future__ import annotations

import html
import pickle
import re
import unicodedata
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


def _normalize_metabolite_token(name: str) -> str:
    t = unicodedata.normalize("NFD", str(name).strip().lower())
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _is_plausible_metabolite_name(name: str) -> bool:
    t = str(name).strip()
    if len(t) < 2:
        return False
    if t.endswith("-OUT"):
        return False
    if t in {"C00000", "***", "**", "*"}:
        return False
    if re.fullmatch(r"C\d{5,}", t):
        return False
    return True


def _token_variants(raw: str) -> set[str]:
    base = _normalize_metabolite_token(raw)
    if not base:
        return set()
    beta = "\u03b2"
    alpha = "\u03b1"
    out = {
        base,
        base.replace(beta, "B").replace(alpha, "A").replace("ß", "ss"),
    }
    if base.startswith("B-") and len(base) > 2:
        out.add(f"{beta}-{base[2:]}")
    if base.startswith(f"{beta}-") and len(base) > 2:
        out.add(f"B-{base[2:]}")
    if "alanine" in base and (base.startswith("B-") or base.startswith(f"{beta}-")):
        out.add("beta-alanine")
    return {x for x in out if x}


def _json_float(v) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def build_metabolite_map_bundle(
    meta: pd.DataFrame | None,
    flux_df: pd.DataFrame | None,
) -> dict | None:
    """
    Curated metabolites from metabolic_model_metadata.csv, enriched with flux rows from df_features
    where reaction strings match. Used by the metabolic map iframe (sidebar list + hover cards).
    """
    need = {"Compound_IN_name", "Compound_OUT_name", "rxnName", "Super.Module.class", "Compound_IN_ID", "Compound_OUT_ID"}
    if meta is None or meta.empty or not need.issubset(meta.columns):
        return None

    fd = pd.DataFrame()
    if flux_df is not None and not flux_df.empty and "feature" in flux_df.columns:
        fd = flux_df.copy()
        fd["_rk"] = fd["feature"].map(normalize_reaction_key)
        fd = fd.drop_duplicates("_rk", keep="first").set_index("_rk", drop=False)

    reaction_importance_rank: dict[str, int] = {}
    if not fd.empty and "mean_rank" in fd.columns:
        for idx in fd.index:
            row = fd.loc[idx]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if "combined_order_mod" in row.index and pd.notna(row["combined_order_mod"]):
                reaction_importance_rank[idx] = int(row["combined_order_mod"])
        if len(reaction_importance_rank) < len(fd):
            sub = fd.sort_values("mean_rank", ascending=True, kind="mergesort")
            for i, idx in enumerate(sub.index, start=1):
                reaction_importance_rank.setdefault(idx, i)

    buckets: dict[str, dict] = {}

    def touch(key: str, display: str) -> dict:
        if key not in buckets:
            buckets[key] = {
                "key": key,
                "name": display.strip(),
                "tokens": set(),
                "chebi": set(),
                "reactions": [],
                "supermodules": set(),
            }
        b = buckets[key]
        b["tokens"].update(_token_variants(display))
        return b

    for _, row in meta.iterrows():
        sub_raw = row["Compound_IN_name"]
        prod_raw = row["Compound_OUT_name"]
        rxn = str(row["rxnName"]).strip()
        rk = normalize_reaction_key(rxn)
        smod = row.get("Super.Module.class")
        smod_s = str(smod).strip() if smod is not None and str(smod) != "nan" else ""

        fr = None
        if rk in fd.index:
            fr = fd.loc[rk]
            if isinstance(fr, pd.DataFrame):
                fr = fr.iloc[0]

        mean_rank = _json_float(fr["mean_rank"]) if fr is not None and "mean_rank" in fr.index else None
        log_fc = _json_float(fr["log_fc"]) if fr is not None and "log_fc" in fr.index else None
        pval_adj = _json_float(fr["pval_adj"]) if fr is not None and "pval_adj" in fr.index else None
        pathway = None
        if fr is not None and "pathway" in fr.index:
            pv = fr["pathway"]
            if pd.notna(pv):
                pathway = str(pv).strip()
        fate_group = None
        if fr is not None and "group" in fr.index:
            g = fr["group"]
            if pd.notna(g):
                fate_group = str(g).strip()

        imp_r = reaction_importance_rank.get(rk)

        base_rx = {
            "reaction": rxn,
            "supermodule": smod_s,
            "mean_rank": mean_rank,
            "importance_rank": imp_r,
            "log_fc": log_fc,
            "pval_adj": pval_adj,
            "pathway": pathway,
            "fate_group": fate_group,
        }

        if _is_plausible_metabolite_name(sub_raw):
            k = _normalize_metabolite_token(sub_raw)
            b = touch(k, str(sub_raw).strip())
            if smod_s:
                b["supermodules"].add(smod_s)
            b["chebi"].add(str(row["Compound_IN_ID"]).strip())
            b["reactions"].append({**base_rx, "as": "substrate", "partner": str(prod_raw).strip()})
        if _is_plausible_metabolite_name(prod_raw):
            k = _normalize_metabolite_token(prod_raw)
            b = touch(k, str(prod_raw).strip())
            if smod_s:
                b["supermodules"].add(smod_s)
            b["chebi"].add(str(row["Compound_OUT_ID"]).strip())
            b["reactions"].append({**base_rx, "as": "product", "partner": str(sub_raw).strip()})

    if not buckets:
        return None

    by_key: dict[str, dict] = {}
    ordered: list[dict] = []

    for key, b in buckets.items():
        seen_rx: set[tuple[str, str]] = set()
        uniq_rx: list[dict] = []
        for r in b["reactions"]:
            sig = (normalize_reaction_key(r["reaction"]), r["as"])
            if sig in seen_rx:
                continue
            seen_rx.add(sig)
            uniq_rx.append(r)
        b["reactions"] = uniq_rx

        imp_ranks = [r["importance_rank"] for r in uniq_rx if r.get("importance_rank") is not None]
        best_importance = min(imp_ranks) if imp_ranks else None

        chebi_sorted = sorted({x for x in b["chebi"] if x and x not in {"nan", "C00000"}})
        tokens_sorted = sorted(b["tokens"])
        smods = sorted(b["supermodules"])

        lines: list[str] = [f"<strong>{html.escape(b['name'])}</strong>"]
        if chebi_sorted:
            lines.append(f"Model IDs: {html.escape(', '.join(chebi_sorted[:8]))}")
        if smods:
            lines.append(f"Modules: {html.escape(' · '.join(smods[:4]))}")
        if best_importance is not None:
            lines.append(f"Strongest linked step: #{best_importance}")

        top_rx = sorted(
            uniq_rx,
            key=lambda r: (
                r.get("importance_rank") is None,
                r["importance_rank"] if r.get("importance_rank") is not None else 10**9,
            ),
        )[:5]
        if top_rx:
            lines.append("<span style='color:#656d76'>Linked reactions (# · log₂FC · fate)</span>")
        for r in top_rx:
            bits = [html.escape(r["reaction"][:80] + ("…" if len(r["reaction"]) > 80 else ""))]
            if r.get("importance_rank") is not None:
                bits.append(f"#{r['importance_rank']}")
            if r["log_fc"] is not None:
                bits.append(f"log₂FC&nbsp;{r['log_fc']:.3f}")
            if r["fate_group"]:
                bits.append(html.escape(r["fate_group"]))
            if r["pathway"]:
                bits.append(f"({html.escape(r['pathway'])})")
            lines.append(" · ".join(bits))

        precursors = sorted(
            {r["partner"] for r in uniq_rx if r["as"] == "product" and r.get("partner") and _is_plausible_metabolite_name(r["partner"])}
        )
        products = sorted(
            {r["partner"] for r in uniq_rx if r["as"] == "substrate" and r.get("partner") and _is_plausible_metabolite_name(r["partner"])}
        )
        if precursors:
            lines.append(
                f"<span style='color:#656d76'>Model precursors (substrates in linked steps)</span><br/>"
                f"{html.escape(', '.join(precursors[:8]))}"
            )
        if products:
            lines.append(
                f"<span style='color:#656d76'>Model products (downstream in linked steps)</span><br/>"
                f"{html.escape(', '.join(products[:8]))}"
            )

        blurb = "<br/>".join(lines)

        search_parts: list[str] = [b["name"], key, *tokens_sorted, *smods, *chebi_sorted]
        for r in uniq_rx:
            search_parts.extend(
                [
                    str(r.get("reaction") or ""),
                    str(r.get("pathway") or ""),
                    str(r.get("fate_group") or ""),
                    str(r.get("supermodule") or ""),
                    str(r.get("as") or ""),
                    str(r.get("partner") or ""),
                ]
            )
            if r.get("importance_rank") is not None:
                search_parts.append(str(r["importance_rank"]))
            if r.get("mean_rank") is not None:
                search_parts.append(str(r["mean_rank"]))
            if r.get("log_fc") is not None:
                search_parts.append(str(r["log_fc"]))
        search_parts.extend(precursors)
        search_parts.extend(products)
        search_text = re.sub(r"\s+", " ", " ".join(search_parts).lower()).strip()

        card = {
            "key": key,
            "name": b["name"],
            "tokens": tokens_sorted,
            "importance_rank": best_importance,
            "n_reactions": len(uniq_rx),
            "blurb_html": blurb,
            "search_text": search_text,
        }
        by_key[key] = card
        ordered.append(card)

    ordered.sort(
        key=lambda c: (
            c["importance_rank"] is None,
            c["importance_rank"] if c["importance_rank"] is not None else 10**9,
            str(c["name"]).lower(),
        )
    )

    return {"list": ordered, "by_key": by_key}


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
