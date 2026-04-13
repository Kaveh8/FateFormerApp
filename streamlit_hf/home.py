"""Landing content for the FateFormer Streamlit hub."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from streamlit_hf.lib import ui

_CACHE = Path(__file__).resolve().parent / "cache"
_HAS_CACHE = (_CACHE / "latent_umap.pkl").is_file() and (_CACHE / "df_features.parquet").is_file()

ui.inject_app_styles()

st.title("FateFormer: interactive analysis")
st.caption("Choose a workspace below or use the sidebar. All views use the same precomputed validation results.")

if not _HAS_CACHE:
    st.warning(
        "This deployment does not have precomputed results yet. Ask the maintainer to publish data, then reload."
    )
else:
    st.success("Precomputed results are available. After a server-side update, refresh the browser to load new plots.")

st.subheader("Open a page")
r1a, r1b, r1c = st.columns(3)
with r1a:
    with st.container(border=True):
        st.page_link("pages/1_Single_Cell_Explorer.py", label="Single-Cell Explorer", icon=":material/scatter_plot:")
        st.caption("Latent UMAP: colour by fate, prediction, fold, batch, modalities, or dominant fate emphasis.")
with r1b:
    with st.container(border=True):
        st.page_link("pages/2_Feature_insights.py", label="Feature Insights", icon=":material/analytics:")
        st.caption("Shift and attention rankings, cohort comparisons, and full feature tables.")
with r1c:
    with st.container(border=True):
        st.page_link("pages/3_Flux_analysis.py", label="Flux Analysis", icon=":material/account_tree:")
        st.caption("Reaction pathways, differential flux, rankings, and model metadata.")
r2a, _, _ = st.columns(3)
with r2a:
    with st.container(border=True):
        st.page_link(
            "pages/4_Gene_expression_analysis.py",
            label="Gene Expression & TF Activity",
            icon=":material/genetics:",
        )
        st.caption("Pathway enrichment, motif activity, and gene / motif tables.")

st.markdown("---")
st.markdown(
    "**Tips:** use chart toolbars for pan/zoom and lasso selection where offered. Tables support search and column sort from the header row."
)
