"""Pathway enrichment tables (DAVID-style exports) for Reactome and KEGG panels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DE_TSV = REPO_ROOT / "analysis" / "de_all_48.tsv"
RE_TSV = REPO_ROOT / "analysis" / "re_all_48.tsv"


def load_de_re_tsv() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if not DE_TSV.is_file() or not RE_TSV.is_file():
        return None
    return pd.read_csv(DE_TSV, sep="\t"), pd.read_csv(RE_TSV, sep="\t")


def preprocess_pathway_file(df: pd.DataFrame, splitter: str) -> pd.DataFrame:
    out = df.copy()
    out["Term"] = out["Term"].astype(str).str.split(splitter).str[-1]
    if splitter == "-":
        out["Term"] = out["Term"].astype(str).str.split("~").str[-1]
    out = out[out["Benjamini"] < 0.05].copy()
    out["Gene Ratio"] = out["Count"] / out["List Total"]
    return out


def merged_reactome_kegg_bubble_frames(
    de_all: pd.DataFrame, re_all: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows for bubble plot (Gene Ratio, Count, Benjamini, Library, Term) per notebook cell 31."""
    reactome_de = de_all[de_all["Category"] == "REACTOME_PATHWAY"]
    reactome_re = re_all[re_all["Category"] == "REACTOME_PATHWAY"]
    kegg_de = de_all[de_all["Category"] == "KEGG_PATHWAY"]
    kegg_re = re_all[re_all["Category"] == "KEGG_PATHWAY"]

    rde = preprocess_pathway_file(reactome_de, "~")
    rde["Library"] = "Reactome"
    rre = preprocess_pathway_file(reactome_re, "~")
    rre["Library"] = "Reactome"
    kde = preprocess_pathway_file(kegg_de, ":")
    kde["Library"] = "KEGG"
    kre = preprocess_pathway_file(kegg_re, ":")
    kre["Library"] = "KEGG"

    merged_dead = pd.concat([rde, kde], ignore_index=True)
    merged_re = pd.concat([rre, kre], ignore_index=True)
    return merged_dead, merged_re


def _preprocess_exploded(df: pd.DataFrame, pval_threshold: float, splitter: str, label: str) -> pd.DataFrame:
    d = df.copy()
    d["Term"] = d["Term"].astype(str).str.split(splitter).str[-1]
    if splitter == "-":
        d["Term"] = d["Term"].astype(str).str.split("~").str[-1]

    def _trunc(x: str) -> str:
        return x[:60] + "..." if len(x) > 60 else x

    d["Term"] = d["Term"].map(_trunc)
    d = d[d["Benjamini"] < pval_threshold]
    sub = d[["Term", "Genes", "Benjamini"]].copy()
    sub["Label"] = label
    exploded = (
        sub.set_index(["Term", "Benjamini", "Label"])["Genes"].str.split(", ").explode().reset_index()
    )
    return exploded


def _binary_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    binary = pd.crosstab(data["Term"], data["Genes"])
    labels = data.groupby("Term")["Label"].first()
    pvals = data.groupby("Term")["Benjamini"].first()
    return binary, labels, pvals


def _sort_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    sp = matrix.sum(axis=1).sort_values(ascending=False).index
    sg = matrix.sum(axis=0).sort_values(ascending=False).index
    return matrix.loc[sp, sg]


def build_merged_pathway_membership(
    de_all: pd.DataFrame, re_all: pd.DataFrame, pval_threshold: float = 0.05
) -> tuple[np.ndarray, list[str], list[str]] | None:
    """
    Numeric grid for heatmap: values 0=white, 1=dead-end gene, 2=reprogramming gene,
    3=Reactome library stripe, 4=KEGG library stripe (notebook cell 29).
    """
    reactome_de = de_all[de_all["Category"] == "REACTOME_PATHWAY"]
    reactome_re = re_all[re_all["Category"] == "REACTOME_PATHWAY"]
    kegg_de = de_all[de_all["Category"] == "KEGG_PATHWAY"]
    kegg_re = re_all[re_all["Category"] == "KEGG_PATHWAY"]

    rde = _preprocess_exploded(reactome_de, pval_threshold, "~", "Dead-end")
    rre = _preprocess_exploded(reactome_re, pval_threshold, "~", "Reprogramming")
    rcomb = pd.concat([rde, rre], ignore_index=True)
    kde = _preprocess_exploded(kegg_de, pval_threshold, ":", "Dead-end")
    kre = _preprocess_exploded(kegg_re, pval_threshold, ":", "Reprogramming")
    kcomb = pd.concat([kde, kre], ignore_index=True)

    rm, rlab, _ = _binary_matrix(rcomb)
    km, klab, _ = _binary_matrix(kcomb)
    rm = _sort_matrix(rm)
    km = _sort_matrix(km)

    reactome_lib = pd.Series("Reactome", index=rm.index)
    kegg_lib = pd.Series("KEGG", index=km.index)
    merged = pd.concat([rm, km], axis=0, sort=False).fillna(0)
    if merged.empty or merged.shape[1] == 0:
        return None
    merged_labels = pd.concat([rlab, klab])
    merged_library = pd.concat([reactome_lib, kegg_lib])

    label_code = {"Dead-end": 1, "Reprogramming": 2}
    lib_code = {"Reactome": 3, "KEGG": 4}

    gene_cols = list(merged.columns)
    z = np.zeros((len(merged), len(gene_cols) + 1), dtype=float)
    for i, term in enumerate(merged.index):
        lc = label_code.get(str(merged_labels.loc[term]), 0)
        for j, g in enumerate(gene_cols):
            v = float(merged.loc[term, g])
            if v > 0 and lc:
                z[i, j] = v * lc
        z[i, -1] = lib_code.get(str(merged_library.loc[term]), 0)

    row_labels = [str(t) for t in merged.index]
    col_labels = gene_cols + ["Library"]
    return z, row_labels, col_labels
