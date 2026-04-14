"""Feature Insights: modality spotlight (RNA, ATAC, Flux)."""

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

_HELP_PAGE = """
**Layout:** Three modality columns (**RNA**, **ATAC**, **Flux**). Each column uses only that modality’s features (**genes**, **TF motifs** from chromatin, or **metabolic reactions**).

**Joint row** (*Joint top markers*): Features ordered by **mean rank** (combined shift + attention; **lower mean rank** = stronger joint priority). Each row is one feature with **two bars** (shift and attention), **min–max scaled within this top‑N list** (0–1) so both are comparable. **Hover** a bar for the full name.

**Shift row** (*Shift importance*): **Shift-only** top **N** by latent shift score per column. **Longer bar** = larger shift in this list. **Hover** for the full name.

**Attention row** (*Attention importance*): **Attention-only** top **N** by rollout importance per column. **Longer bar** = more average attention. **Hover** for the full name.
"""

df = io.load_df_features()

if df is None:
    st.error(
        "Feature data are not loaded. Ask the maintainer to publish results for this app, then reload."
    )
    st.stop()

st.title(ui.FEATURE_INSIGHTS_TITLE)
st.caption(ui.FEATURE_INSIGHTS_CAPTION)
try:
    _spot_h_l, _spot_h_r = st.columns([0.94, 0.06], gap="small", vertical_alignment="center")
except TypeError:
    _spot_h_l, _spot_h_r = st.columns([0.94, 0.06], gap="small")
with _spot_h_l:
    st.subheader("Modality spotlight")
with _spot_h_r:
    ui.plot_help_popover(_HELP_PAGE, key="t2_page_help")
st.caption(
    "Here, we zoom into one modality at a time (RNA, ATAC, or Flux) to explore top fate predictor markers: for each column "
    "you see joint top markers, then shift-only and attention-only rankings side by side so within-modality comparisons "
    "stay fair."
)
top_n_rank = st.slider(
    "Top N per chart",
    10,
    55,
    20,
    key="t2_topn",
    help=(
        "Number of features in each chart on this page: the joint (mean-rank) row, the shift-only row, "
        "and the attention-only row all use this N within each modality column."
    ),
)

st.markdown("##### Joint top markers (by mean rank)")
st.caption(
    "Joint row: strongest by mean rank; shift and attention bars scaled within this top-N list. Hover a bar for the full name."
)
r1a, r1b, r1c = st.columns(3)
for col, mod in zip((r1a, r1b, r1c), ("RNA", "ATAC", "Flux")):
    sm = df[df["modality"] == mod]
    if sm.empty:
        continue
    with col:
        st.plotly_chart(
            plots.joint_shift_attention_top_features(sm, mod, top_n_rank),
            width="stretch",
        )

st.markdown("##### Shift importance")
r2a, r2b, r2c = st.columns(3)
for col, mod in zip((r2a, r2b, r2c), ("RNA", "ATAC", "Flux")):
    sm = df[df["modality"] == mod]
    if sm.empty:
        continue
    colc = plots.MODALITY_COLOR.get(mod, plots.PALETTE[0])
    sub = sm.nlargest(top_n_rank, "importance_shift").sort_values("importance_shift", ascending=True)
    with col:
        st.plotly_chart(
            plots.rank_bar(
                sub,
                "importance_shift",
                "feature",
                f"{mod}: shift · top {top_n_rank}",
                colc,
                xaxis_title="Latent shift importance",
            ),
            width="stretch",
        )

st.markdown("##### Attention importance")
r3a, r3b, r3c = st.columns(3)
for col, mod in zip((r3a, r3b, r3c), ("RNA", "ATAC", "Flux")):
    sm = df[df["modality"] == mod]
    if sm.empty:
        continue
    colc = plots.MODALITY_COLOR.get(mod, plots.PALETTE[0])
    sub = sm.nlargest(top_n_rank, "importance_att").sort_values("importance_att", ascending=True)
    with col:
        st.plotly_chart(
            plots.rank_bar(
                sub,
                "importance_att",
                "feature",
                f"{mod}: attention · top {top_n_rank}",
                colc,
                xaxis_title="Attention importance",
            ),
            width="stretch",
        )
