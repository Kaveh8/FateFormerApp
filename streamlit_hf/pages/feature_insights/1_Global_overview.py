"""Feature Insights: global overview of multimodal feature importance."""

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
**What this is:** The **top important fate-predictor markers** for **FateFormer** across its **three modalities** (**RNA** genes, **TF motifs** from chromatin (ATAC), and **flux** reactions), as a **global** view that combines **latent shift** probes and **attention rollout** over the full model.

**Panels:** **Shift** and **attention** bar charts show the **top‑N** features for each metric (**min‑max scaled within that chart**, longest bar = 1). The **pie chart** (right) shows **modality mix** (RNA vs ATAC vs Flux) among a pool of **strongest** features by **mean rank** (**lower mean rank** = higher joint priority).

**How to read it:** **Longer bars** mean stronger measured influence for that metric. **Colours** mark **modality**. Use the **sliders** above to change bar count and pie pool size. The **pie chart** answers: “Among the most important features in this pool, which data type dominates?”.
"""

df = io.load_df_features()

if df is None:
    st.error(
        "Feature data are not loaded. Ask the maintainer to publish results for this app, then reload."
    )
    st.stop()

st.title(ui.FEATURE_INSIGHTS_TITLE)
st.caption(ui.FEATURE_INSIGHTS_CAPTION)
st.subheader("Global overview")
st.caption(
    "Here, we give a birds-eye view of which RNA, ATAC, and Flux features matter most: top-N bars for latent shift and "
    "attention (two explainability methods), plus a pie of modality mix among the strongest features by mean rank "
    "(sliders change list sizes)."
)
c1, c2 = st.columns(2)
with c1:
    top_n_bars = st.slider(
        "Top N (shift & attention bars)",
        10,
        45,
        20,
        key="t1_topn_bars",
        help=(
            "How many features appear in the left (latent shift) and middle (attention) bar charts: the top N by each "
            "metric. Each chart is min–max scaled on its own (longest bar = 1). Increase N to list more markers; "
            "decrease N to focus on the strongest few."
        ),
    )
with c2:
    top_n_pie = st.slider(
        "Pool size (mean-rank pie)",
        50,
        250,
        100,
        key="t1_topn_pie",
        help=(
            "How many features define the right-hand pie chart: the N strongest by mean rank (lower mean rank = "
            "stronger joint ranking across shift and attention). A larger pool gives a broader modality mix "
            "(RNA vs ATAC vs Flux); a smaller pool weights only the very top joint features."
        ),
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
