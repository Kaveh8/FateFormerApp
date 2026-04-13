"""Gene expression — searchable motif / TF table."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from streamlit_hf.lib import io
from streamlit_hf.lib import ui

ui.inject_app_styles()

TABLE_COLS = [
    "mean_rank",
    "feature",
    "rank_shift_in_modal",
    "rank_att_in_modal",
    "combined_order_mod",
    "rank_shift",
    "rank_att",
    "importance_shift",
    "importance_att",
    "top_10_pct",
    "mean_de",
    "mean_re",
    "group",
    "log_fc",
    "pval_adj",
    "mean_diff",
    "pval_adj_log",
]


def _table_cols(show: pd.DataFrame) -> list[str]:
    return [c for c in TABLE_COLS if c in show.columns]


st.title("Gene Expression & TF Activity")
st.caption(
    "Pathway enrichment (Reactome / KEGG) and a pathway-gene map; chromVAR-style motif deviations and activity by "
    "fate; sortable gene and motif tables. Use **Feature Insights** for global shift and attention rankings across modalities."
)

df = io.load_df_features()
if df is None:
    st.error("Feature data could not be loaded. Reload after results are published, or contact the maintainer.")
    st.stop()

rna = df[df["modality"] == "RNA"].copy()
atac = df[df["modality"] == "ATAC"].copy()
if rna.empty and atac.empty:
    st.warning("No RNA gene or ATAC motif features are available in the current results.")
    st.stop()

st.subheader("Motif table")
if atac.empty:
    st.warning("No motif-level ATAC features are available in the current results.")
else:
    q = st.text_input("Filter by motif or TF", "", key="tf_tbl_q")
    show = atac
    if q.strip():
        show = show[show["feature"].astype(str).str.contains(q, case=False, na=False)]
    cols = _table_cols(show)
    st.dataframe(show[cols].sort_values("mean_rank"), width="stretch", hide_index=True)
    st.download_button(
        "Download table (CSV)",
        show[cols].sort_values("mean_rank").to_csv(index=False).encode("utf-8"),
        file_name="tf_motif_table.csv",
        mime="text/csv",
        key="tf_tbl_dl",
    )
