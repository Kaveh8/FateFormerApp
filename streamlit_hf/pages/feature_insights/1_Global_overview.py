"""Feature Insights — global overview of multimodal feature importance."""

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

_GLOBAL_OVERVIEW_HELP = """
**What this is:** A **global** snapshot of which **genes, ATAC peaks, or flux reactions** rank highest when **latent shift probes** and **attention rollout** are combined across the whole model.

**Panels:** **Shift** and **attention** bar charts show the **top‑N** features for each metric (**min‑max scaled within that chart**). The **pie** shows the **RNA / ATAC / Flux** breakdown among a larger pool of **lowest mean‑rank** features (strongest overall joint ranking).

**How to read it:** **Lower mean rank** = higher priority in the joint ranking. **Colours** encode **modality**. Use the sliders to change how many bars and how large the pie pool is.

**Takeaway:** See whether interpretability is **RNA‑heavy**, **metabolism‑heavy**, or **balanced** before drilling into modality pages.
"""

st.title("Feature Insights")
st.caption("Latent-shift probes, attention rollout, and combined rankings across RNA, ATAC, and Flux.")

df = io.load_df_features()

if df is None:
    st.error(
        "Feature data are not loaded. Ask the maintainer to publish results for this app, then reload."
    )
    st.stop()

st.subheader("Global overview")
c1, c2 = st.columns(2)
with c1:
    top_n_bars = st.slider(
        "Top N (shift & attention bars)",
        10,
        45,
        20,
        key="t1_topn_bars",
    )
with c2:
    top_n_pie = st.slider(
        "Pool size (mean-rank pie)",
        50,
        250,
        100,
        key="t1_topn_pie",
    )
ui.plot_caption_with_help(
    "Global top features by shift vs attention; pie = modality mix among strongest mean-rank pool.",
    _GLOBAL_OVERVIEW_HELP,
    key="fi_go_plot_help",
)
st.plotly_chart(
    plots.global_rank_triple_panel(df, top_n=top_n_bars, top_n_pie=top_n_pie),
    width="stretch",
)
