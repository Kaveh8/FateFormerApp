"""Feature Insights: full ranked feature table."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from streamlit_hf.lib import io
from streamlit_hf.lib import ui

ui.inject_app_styles()

_FULL_TABLE_HELP = """
**What this is:** The **full FateFormer ranked feature list** (RNA genes, ATAC TF motifs, flux reactions) with **shift**, **attention**, and **joint** rank columns from the interpretability pipeline.

**Key columns:** **mean_rank** (lower = stronger overall), **rank_shift** / **rank_att** (global), modality‑internal ranks, and **importance_*** scores. Where available, **pathway** / **module** annotate flux or gene context.

**How to use:** **Sort** or **search** in the table toolbar; **download CSV** for spreadsheets or supplementary tables.
"""

df = io.load_df_features()

if df is None:
    st.error(
        "Feature data are not loaded. Ask the maintainer to publish results for this app, then reload."
    )
    st.stop()

st.title(ui.FEATURE_INSIGHTS_TITLE)
st.caption(ui.FEATURE_INSIGHTS_CAPTION)
st.subheader("Full table")
st.caption(
    "Here is the complete ranked feature table for the run (RNA genes, ATAC motifs, flux reactions): every shift, "
    "attention, and joint rank and score the pipeline emitted."
)
scope = st.radio(
    "Table scope",
    ["All modalities", "Single modality"],
    horizontal=True,
    key="t5_scope",
)
mod_tbl = "all"
if scope == "Single modality":
    mod_tbl = st.selectbox("Modality", ["RNA", "ATAC", "Flux"], key="t5_mod")
    tbl = df[df["modality"] == mod_tbl].copy()
else:
    tbl = df.copy()
show_cols = [
    c
    for c in [
        "mean_rank",
        "feature",
        "modality",
        "rank_shift_in_modal",
        "rank_att_in_modal",
        "combined_order_mod",
        "rank_shift",
        "rank_att",
        "importance_shift",
        "importance_att",
        "top_10_pct",
        "group",
        "log_fc",
        "pval_adj",
        "pathway",
        "module",
    ]
    if c in tbl.columns
]
ui.plot_caption_with_help(
    "Full FateFormer list for the chosen scope, sorted by **mean rank** (lower = stronger joint priority).",
    _FULL_TABLE_HELP,
    key="t5_table_help",
)
full_view = tbl[show_cols].sort_values("mean_rank")
st.dataframe(full_view, width="stretch", hide_index=True)
suffix = mod_tbl if scope == "Single modality" else "all"
st.download_button(
    "Download table (CSV)",
    full_view.to_csv(index=False).encode("utf-8"),
    file_name=f"fateformer_features_{suffix}.csv",
    mime="text/csv",
    key="t5_dl",
)
