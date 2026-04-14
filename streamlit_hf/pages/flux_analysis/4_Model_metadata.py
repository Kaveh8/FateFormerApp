"""Flux Analysis: scFEA metabolic model metadata table."""

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

_SCFEA_PMC = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8494226/"
_SCFEA_GITHUB = "https://github.com/changwn/scFEA"

_HELP_MODEL_META = f"""
**Source:** The **metabolic model metadata** from **scFEA** (single-cell flux estimation from scRNA-seq) that is used for inferring flux reactions from scRNA-seq data. Open access article: [{_SCFEA_PMC}]({_SCFEA_PMC}) (*Genome Research*, 2021). Code and model resources: [{_SCFEA_GITHUB}]({_SCFEA_GITHUB}).

**What this is:** The **scFEA** metabolic model info used for inferring fluxomic data from scRNA-seq (one row per substrate → product reaction).
"""

st.title("Flux Analysis")
st.caption(
    "**Flux Analysis** ties inferred **reaction flux** to **pathways**, **fate contrasts**, **rankings**, and **model** metadata. "
    "For multimodal **shift**/**attention** summaries, open **Feature Insights**."
)

meta = io.load_metabolic_model_metadata()

st.subheader("Metabolic model metadata")
st.caption(
    f"Here is the scFEA metabolic model metadata used to interpret flux features: modules, compounds, and reaction names. "
    f"[Paper]({_SCFEA_PMC}), [GitHub]({_SCFEA_GITHUB})."
)
if meta is None or meta.empty:
    st.error("Metabolic model metadata is not available in this build.")
else:
    try:
        _mm_l, _mm_r = st.columns([0.94, 0.06], gap="small", vertical_alignment="center")
    except TypeError:
        _mm_l, _mm_r = st.columns([0.94, 0.06], gap="small")
    with _mm_r:
        ui.plot_help_popover(_HELP_MODEL_META, key="flux_model_meta_help")
    sm_ids = sorted(meta["Supermodule_id"].dropna().unique().astype(int).tolist())
    graph_labels = ["All modules"]
    for sid in sm_ids:
        cls = str(meta.loc[meta["Supermodule_id"] == sid, "Super.Module.class"].iloc[0])
        graph_labels.append(f"{sid}: {cls}")
    tix = st.selectbox(
        "Model scope",
        range(len(graph_labels)),
        format_func=lambda i: graph_labels[i],
        key="flux_model_scope",
        help=(
            "**All modules:** every edge row in the metadata CSV. **Named supermodule:** only edges with that "
            "**Supermodule_id** (class label shown in the menu)."
        ),
    )
    supermodule_id = None if tix == 0 else sm_ids[tix - 1]
    tbl = io.build_metabolic_model_table(meta, supermodule_id=supermodule_id)
    st.dataframe(tbl, width="stretch", hide_index=True)
    st.download_button(
        "Download metabolic model metadata (CSV)",
        tbl.to_csv(index=False).encode("utf-8"),
        file_name="fateformer_metabolic_model_edges.csv",
        mime="text/csv",
        key="flux_model_dl",
        help="CSV export of the table above for the current **Model scope**.",
    )
