"""Flux Analysis — pathway sunburst and reaction annotation panels."""

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

_HELP_FLUX_SUNBURST = """
**What this is:** A **hierarchical view** of **metabolic pathways** and the **individual flux reactions** that rank highest by **mean importance** in this model.

**How to read it:** **Inner rings** = pathway context; **outer segments** = **reactions**. Larger / more central emphasis (depends on layout) highlights **stronger combined ranking** in the results table. Use the slider to include more or fewer reactions.

**Takeaway:** Quickly see **which pathways dominate** the model’s flux interpretation layer.
"""

_HELP_FLUX_ANNOTATION = """
**What this is:** **Heatmaps** aligned to the **same top reactions** as the sunburst: each row is a **reaction**, columns summarise **pathway membership**, **differential flux** (Log₂ fold change between fate groups), and **statistical significance**.

**How to read it:** Scan rows for reactions that are both **statistically notable** and **highly ranked** by the model. **Hover** cells for exact values where Plotly provides tooltips.

**Takeaway:** Links **statistics on measured flux** to **model-derived importance**.
"""

_HELP_FLUX_PROFILE = """
**What this is:** A compact **profile** of **model‑centric metrics** (e.g. joint ranks) for the same **top reactions**, complementary to the heatmaps.

**How to read it:** Compare **relative bars/scores** across reactions—**longer** usually means **stronger model priority** for that reaction in this summary.

**Takeaway:** A second lens that tracks **interpretability scores** rather than raw flux alone.
"""

st.title("Flux Analysis")
st.caption(
    "Reaction-level flux: how pathways, statistics, and model rankings line up. "
    "For global rank bars and shift vs. attention scatter, open **Feature insights**."
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

st.subheader("Pathway map")
if not _data_ok:
    st.error(_data_msg)
else:
    st.caption(
        "**Left:** sunburst of the strongest reactions by mean rank, grouped by pathway. **Right:** heatmaps for the "
        "same reactions: pathway, differential Log₂FC, and statistical significance, aligned row by row. "
        "Ranked reaction table: **Reaction ranking**. Curated model edges: **Model metadata**."
    )
    try:
        c1, c2 = st.columns([1.05, 0.95], gap="medium", vertical_alignment="top")
    except TypeError:
        c1, c2 = st.columns([1.05, 0.95], gap="medium")
    with c1:
        n_sb = st.slider("Reactions in sunburst", 25, 90, 52, key="flux_sb_n")
        _, _hp = st.columns([1, 0.22])
        with _hp:
            ui.plot_help_popover(_HELP_FLUX_SUNBURST, key="flux_sb_help")
        st.plotly_chart(plots.flux_pathway_sunburst(flux, max_features=n_sb), width="stretch")
    with c2:
        top_n_nb = st.slider("Reactions in annotation + profile", 12, 40, 26, key="flux_nb_n")
        _, _hp = st.columns([1, 0.22])
        with _hp:
            ui.plot_help_popover(_HELP_FLUX_ANNOTATION, key="flux_ann_help")
        st.plotly_chart(
            plots.flux_reaction_annotation_panel(flux, top_n=top_n_nb, metric="mean_rank"),
            width="stretch",
        )
        _, _hp2 = st.columns([1, 0.22])
        with _hp2:
            ui.plot_help_popover(_HELP_FLUX_PROFILE, key="flux_prof_help")
        st.plotly_chart(
            plots.flux_model_metric_profile(flux, top_n=min(top_n_nb, 24), metric="mean_rank"),
            width="stretch",
        )
