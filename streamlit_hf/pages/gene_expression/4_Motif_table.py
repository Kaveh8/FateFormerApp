"""Gene expression: searchable motif / TF table."""

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

_HELP_MOTIF_TABLE = """
**ATAC** motif / TF features used in this run: **one row per feature**, sorted by **mean rank**. Columns include **FateFormer** ranking and attribution, **per-fate** activity summaries, and **differential** statistics. Search to narrow the list; use **Download** for a CSV copy.
"""

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
    "**Pathways** (Reactome / KEGG) and pathway–gene views; **ATAC motif** deviation and **TF activity** by fate; "
    "**gene** and **motif** tables."
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

try:
    _mt_h_l, _mt_h_r = st.columns([0.94, 0.06], gap="small", vertical_alignment="center")
except TypeError:
    _mt_h_l, _mt_h_r = st.columns([0.94, 0.06], gap="small")
with _mt_h_l:
    st.subheader("Motif table")
with _mt_h_r:
    ui.plot_help_popover(_HELP_MOTIF_TABLE, key="ge_motif_table_help")
st.caption(
    "Here is a searchable table of all ATAC motif / TF features, each with FateFormer ranks and per-fate activity and "
    "differential fields that you can sort, filter by name, or download CSV."
)

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
