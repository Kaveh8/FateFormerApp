"""Flux Analysis: pathway sunburst and reaction annotation panels."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from streamlit_hf.lib import io
from streamlit_hf.lib import plots
from streamlit_hf.lib import ui

ui.inject_app_styles()

_HELP_PATHWAY_MAP = """
**Layout:** **Left column:** **sunburst**. **Right column:** **Pathway / Log₂FC / significance** (three **heatmap** columns, one **row** per reaction).

**Sunburst:** **Inner ring** = **pathway**; **outer ring** = **reaction**. Reactions are the top set by **mean_rank** (FateFormer joint rank; **lower** = stronger). **Wedge size** reflects that ranking. **Colour** = per-reaction **log₂ fold change** in inferred flux for **reprogramming** vs **dead-end** samples (experimental labels).

**Pathway / Log₂FC / significance:** Same top-**N** reactions as the **Reactions in heatmap** slider (**N** rows). **Columns:** **Pathway** (categorical colour), **Log₂FC** (reprogramming vs dead-end), **−log₁₀ adjusted p** for that contrast. **Hover** for exact values.

**Sliders:** **Reactions in sunburst** adjusts only the **left** sunburst. **Reactions in heatmap** sets how many top reactions appear in the **right-hand** heatmap.
"""

st.title("Flux Analysis")
st.caption(
    "**Flux Analysis** ties inferred **reaction flux** to **pathways**, **fate contrasts**, **rankings**, and **model** metadata. "
    "For multimodal **shift**/**attention** summaries, open **Feature Insights**."
)

try:
    df = io.load_df_features()
except Exception:
    df = None

_data_ok = True
if df is None:
    _data_ok = False
    _data_msg = (
        "Flux and feature data are not loaded in this session. Reload the app after the maintainer has published "
        "fresh results, or ask them to check the deployment."
    )
    flux = None
else:
    flux = df[df["modality"] == "Flux"].copy()
    if flux.empty:
        _data_ok = False
        _data_msg = "There are no flux reactions in the current results."
        flux = None

try:
    _pm_h_l, _pm_h_r = st.columns([0.94, 0.06], gap="small", vertical_alignment="center")
except TypeError:
    _pm_h_l, _pm_h_r = st.columns([0.94, 0.06], gap="small")
with _pm_h_l:
    st.subheader("Pathway map")
with _pm_h_r:
    ui.plot_help_popover(_HELP_PATHWAY_MAP, key="flux_pathway_map_help")

if not _data_ok:
    st.error(_data_msg)
else:
    st.caption(
        "Here, we map top FateFormer-ranked flux reactions into pathway context: a sunburst (pathway → reaction) and a "
        "heatmap of pathway, log₂ fold change, and significance for reprogramming vs dead-end."
    )
    try:
        c1, c2 = st.columns([1.05, 0.95], gap="medium", vertical_alignment="top")
    except TypeError:
        c1, c2 = st.columns([1.05, 0.95], gap="medium")
    with c1:
        n_sb = st.slider(
            "Reactions in sunburst",
            25,
            90,
            52,
            key="flux_sb_n",
            help=(
                "How many **top** flux reactions (by **mean rank**) appear in the **sunburst** only. "
                "Does not change the heatmap; use the other slider for that."
            ),
        )
        st.plotly_chart(plots.flux_pathway_sunburst(flux, max_features=n_sb), width="stretch")
    with c2:
        top_n_nb = st.slider(
            "Reactions in heatmap",
            12,
            40,
            26,
            key="flux_nb_n",
            help=(
                "How many **top** flux reactions (by **mean rank**) appear as **rows** in the **Pathway / Log₂FC / significance** heatmap."
            ),
        )
        st.plotly_chart(
            plots.flux_reaction_annotation_panel(flux, top_n=top_n_nb, metric="mean_rank"),
            width="stretch",
        )
