"""Gene expression — Reactome / KEGG pathway enrichment."""

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

_HELP_PATH_BUBBLE_DE = """
**What this is:** **Pathway over‑representation** among genes linked to **dead‑end** cells (Reactome + KEGG merged view). **Significance** is **Benjamini–Hochberg FDR** (*q* < 0.05).

**How to read it:** Each **bubble** is a pathway; **position** reflects effect size / enrichment strength; **size** often tracks **gene count** or **significance** (see axis labels and hover). Compare to the **reprogramming** panel for fate‑specific patterns.

**Takeaway:** Highlights **process‑level** themes in the dead‑end transcriptional state.
"""

_HELP_PATH_BUBBLE_RE = """
**What this is:** The same **enrichment style** as dead‑end, but for genes associated with **reprogramming** outcomes.

**How to read it:** Interpret **bubble position and size** as in the dead‑end panel. Pathways **strong here but not there** (and vice‑versa) are the most **discriminating**.

**Takeaway:** Complements RNA‑level interpretability with **known pathway databases**.
"""

_HELP_PATH_HEAT = """
**What this is:** A **gene × pathway** **heatmap** of **membership** among **leading** genes from the enrichment results (Reactome / KEGG). **Empty** cells mean no assignment in that slice of the matrix.

**How to read it:** **Rows** = genes; **columns** = pathways. **Colour intensity** shows presence/strength of membership depending on the encoding (use **hover**).

**Takeaway:** Moves from **pathway lists** to a **literal gene‑to‑pathway map** for follow‑up.
"""

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

st.subheader("Gene pathway enrichment")
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
        _, _hp = st.columns([1, 0.22])
        with _hp:
            ui.plot_help_popover(_HELP_PATH_BUBBLE_DE, key="ge_bubble_de_help")
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
        _, _hp = st.columns([1, 0.22])
        with _hp:
            ui.plot_help_popover(_HELP_PATH_BUBBLE_RE, key="ge_bubble_re_help")
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
        _, _hp = st.columns([1, 0.18])
        with _hp:
            ui.plot_help_popover(_HELP_PATH_HEAT, key="ge_path_heat_help")
        st.plotly_chart(plots.pathway_gene_membership_heatmap(z, ylabs, xlabs), width="stretch")
