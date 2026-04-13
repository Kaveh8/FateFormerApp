"""Flux Analysis — differential flux and fate scatter."""

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
**What this is:** A **volcano plot** for **reaction‑level flux**: **horizontal axis** = differential activity (**Log₂ fold change** between fate groups); **vertical axis** = **statistical significance** (\u2212log\u2081\u2080 **adjusted p**).

**How to read it:** Points **far right/left** change most between groups; points **higher up** are more significant. **Colour** encodes the reaction’s **overall mean rank** in the interpretability table. Unreliable points with **no fold change** and **zero** adjusted p‑value are **dropped**.

**Takeaway:** Highlights reactions that are both **biologically different** and **interpretable** in the model.
"""

_HELP_FLUX_FATE_SCATTER = """
**What this is:** Each **point** is a **flux reaction**. **X** = **average flux** in cells called **dead‑end**; **Y** = average in **reprogramming** cells (per the experimental grouping used in the analysis).

**How to read it:** Points **above the diagonal** are higher in reprogramming; **below** = higher in dead‑end. **Point size** reflects **combined shift + attention** strength; **colour** = **pathway** (minor categories grouped as *Other*).

**Takeaway:** Links **raw flux behaviour** to **model emphasis** (size) and **pathway context** (colour).
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

st.subheader("Differential & fate")
if not _data_ok:
    st.error(_data_msg)
else:
    st.caption(
        "**Volcano:** differential Log₂FC versus significance (\u2212log\u2081\u2080 adjusted p); colour shows overall mean rank. "
        "Points with essentially no fold change and a zero adjusted p-value are removed as unreliable. "
        "**Scatter:** average measured flux in dead-end versus reprogramming cells; point size reflects combined shift "
        "and attention strength; colours mark pathway (largest groups shown, others grouped as *Other*)."
    )
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
