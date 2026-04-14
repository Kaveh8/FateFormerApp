"""Gene expression: Reactome / KEGG pathway enrichment."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from streamlit_hf.lib import io
from streamlit_hf.lib import pathways as pathway_data
from streamlit_hf.lib import plots
from streamlit_hf.lib import ui

ui.inject_app_styles()

_HELP_PATHWAY_ENRICHMENT = """
**Overview:** **Gene pathway enrichment**: Reactome and KEGG **over-representation** from fate-split **RNA marker** lists, then a **pathway × gene** heatmap of the leading hits.

**Bubble panels (dead-end vs reprogramming):** **Leading genes** are **grouped by fate** (dead-end vs reprogramming); each panel runs enrichment on that gene set. **Horizontal axis** = **gene ratio** (enrichment table). **Circles** = **Reactome** pathways; **squares** = **KEGG** pathways. **Vertical** position orders pathways; **size** reflects **gene count**; **colour** = **−log₁₀** Benjamini *q* (*q* < 0.05). **Hover** for pathway name, library, count, and *q*. **Compare** left and right panels for cohort-specific pathways.

**Heatmap:** **Rows** = enriched **pathway terms** (Reactome block, then KEGG). **Columns** = **genes** (from the same fate-split marker lists that fed enrichment) plus a **Library** stripe (**Reactome** vs **KEGG** per row). **Colour** encodes **dead-end** vs **reprogramming** membership for that gene–pathway pair (and the library stripe); **hover** for the exact label. **Empty** cells = no link in this matrix slice.
"""

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
    _pe_h_l, _pe_h_r = st.columns([0.94, 0.06], gap="small", vertical_alignment="center")
except TypeError:
    _pe_h_l, _pe_h_r = st.columns([0.94, 0.06], gap="small")
with _pe_h_l:
    st.subheader("Gene pathway enrichment")
with _pe_h_r:
    ui.plot_help_popover(_HELP_PATHWAY_ENRICHMENT, key="ge_pathway_page_help")
st.caption(
    "Here, we turn fate-split RNA gene markers into Reactome and KEGG over-representation (bubble panels per cohort), "
    "then lay out a pathway × gene heatmap for the leading hits."
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
