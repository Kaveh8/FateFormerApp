"""Flux Analysis: differential flux and fate scatter."""

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

_HELP_FLUX_VOLCANO = """
**What this is:** One **point** per **flux reaction**. **X** = **log₂ fold change** in inferred flux between **dead-end**-labeled and **reprogramming**-labeled samples. **Y** = **−log₁₀ adjusted p-value** for that contrast (multiple-testing adjusted in the results table).

**How to read it:** **Further from zero on X** = stronger shift between cohorts. **Higher on Y** = stronger statistical evidence. **Colour** = **mean rank** (FateFormer joint rank across the feature table; **lower** rank = stronger overall model focus). Reactions with **~zero** fold change and an **adjusted p of exactly zero** are removed as numerical artifacts.

**Hover** the points for reaction name, pathway, and related fields.
"""

_HELP_FLUX_FATE_SCATTER = """
**What this is:** One **point** per **flux reaction**. **X** = **mean flux** across samples labeled **dead-end**; **Y** = **mean flux** across samples labeled **reprogramming** (same per-sample fate labels as elsewhere in this analysis).

**How to read it:** The **y = x** line would mark equal average flux in both cohorts. **Above** the diagonal, average flux is **higher in reprogramming** than in dead-end for that reaction; **below**, **higher in dead-end**. **Marker size** scales with **√(latent shift importance × attention importance)** (capped for display). **Colour** = **pathway**; smaller pathway groups are merged into **Other**.

**Hover** for reaction name, **mean rank**, **log₂FC**, and pathway.
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

st.subheader("Differential & fate")
st.caption(
    "Here, we contrast dead-end and reprogramming cells at the reaction level: a volcano of flux log₂FC vs significance, "
    "and a scatter of mean flux in each cohort with pathway colouring."
)
if not _data_ok:
    st.error(_data_msg)
else:
    b1, b2 = st.columns(2)
    with b1:
        _, _hp = st.columns([1, 0.22])
        with _hp:
            ui.plot_help_popover(_HELP_FLUX_VOLCANO, key="flux_vol_help")
        st.plotly_chart(plots.flux_volcano(flux), width="stretch")
    with b2:
        _, _hp = st.columns([1, 0.22])
        with _hp:
            ui.plot_help_popover(_HELP_FLUX_FATE_SCATTER, key="flux_sc_help")
        st.plotly_chart(plots.flux_dead_end_vs_reprogram_scatter(flux), width="stretch")
