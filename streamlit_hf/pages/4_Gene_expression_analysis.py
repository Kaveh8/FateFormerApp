"""Gene expression and TF motif activity: pathway enrichment, chromVAR-style motifs, and tables."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from streamlit_hf.lib import io
from streamlit_hf.lib import pathways as pathway_data
from streamlit_hf.lib import plots
from streamlit_hf.lib import ui

ui.inject_app_styles()

st.title("Gene Expression & TF Activity")

df = io.load_df_features()
if df is None:
    st.error("Feature data could not be loaded. Reload after results are published, or contact the maintainer.")
    st.stop()

rna = df[df["modality"] == "RNA"].copy()
atac = df[df["modality"] == "ATAC"].copy()
if rna.empty and atac.empty:
    st.warning("No RNA gene or ATAC motif features are available in the current results.")
    st.stop()

st.caption(
    "Pathway enrichment (Reactome / KEGG) and a pathway-gene map; chromVAR-style motif deviations and activity by "
    "fate; sortable gene and motif tables. Use **Feature Insights** for global shift and attention rankings across modalities."
)

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


tab_path, tab_motif, tab_gene_tbl, tab_motif_tbl = st.tabs(
    ["Gene Pathway Enrichment", "Motif Activity", "Gene Table", "Motif Table"]
)

with tab_path:
    st.caption(
        "Over-representation of Reactome and KEGG pathways (Benjamini-Hochberg *q* < 0.05). "
        "The lower panel maps leading genes to pathways; empty grid positions are left clear."
    )
    raw = pathway_data.load_de_re_tsv()
    if raw is None:
        st.info("Pathway enrichment views are not available in this deployment.")
    else:
        de_all, re_all = raw
        mde, mre = pathway_data.merged_reactome_kegg_bubble_frames(de_all, re_all)
        bubble_h = max(
            plots.pathway_bubble_suggested_height(len(mde)),
            plots.pathway_bubble_suggested_height(len(mre)),
        )
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.plotly_chart(
                plots.pathway_enrichment_bubble_panel(
                    mde,
                    "Pathway enrichment: dead-end",
                    show_colorbar=True,
                    layout_height=bubble_h,
                ),
                width="stretch",
            )
        with c2:
            st.plotly_chart(
                plots.pathway_enrichment_bubble_panel(
                    mre,
                    "Pathway enrichment: reprogramming",
                    show_colorbar=True,
                    layout_height=bubble_h,
                ),
                width="stretch",
            )
        hm = pathway_data.build_merged_pathway_membership(de_all, re_all)
        if hm is None:
            st.info("No pathway-gene matrix could be built from the current enrichment results.")
        else:
            z, ylabs, xlabs = hm
            st.plotly_chart(plots.pathway_gene_membership_heatmap(z, ylabs, xlabs), width="stretch")

with tab_motif:
    if atac.empty:
        st.warning("No motif-level ATAC features are available in the current results.")
    else:
        st.caption(
            "Left: mean motif score difference (reprogramming − dead-end) versus significance. "
            "Right: mean activity in each fate; colour and size follow the same encoding as in **Feature Insights**."
        )
        a1, a2 = st.columns(2, gap="medium")
        with a1:
            st.plotly_chart(plots.motif_chromvar_volcano(atac), width="stretch")
        with a2:
            st.plotly_chart(
                plots.notebook_style_activity_scatter(
                    atac,
                    title="TF activity (z-score) by fate",
                    x_title="Dead-end (TF activity)",
                    y_title="Reprogramming (TF activity)",
                ),
                width="stretch",
            )

with tab_gene_tbl:
    if rna.empty:
        st.warning("No RNA gene features are available in the current results.")
    else:
        q = st.text_input("Filter by gene name", "", key="ge_tbl_q")
        show = rna
        if q.strip():
            show = show[show["feature"].astype(str).str.contains(q, case=False, na=False)]
        cols = _table_cols(show)
        st.dataframe(show[cols].sort_values("mean_rank"), width="stretch", hide_index=True)
        st.download_button(
            "Download table (CSV)",
            show[cols].sort_values("mean_rank").to_csv(index=False).encode("utf-8"),
            file_name="gene_expression_table.csv",
            mime="text/csv",
            key="ge_tbl_dl",
        )

with tab_motif_tbl:
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
