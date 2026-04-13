"""Feature Insights — modality spotlight (RNA, ATAC, Flux)."""

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

_HELP_JOINT = """
**What this is:** Within **{mod}** only, features with the **strongest joint ranking** (combined shift + attention priority).

**How to read it:** Each row is **one feature**; the **two bars** are **shift** and **attention** scores **rescaled0–1 within this top‑N list** so they are comparable. **Hover** for the full name.

**Takeaway:** Highlights markers that are important both to **representations** and to **model focus** in this modality.
"""

_HELP_SHIFT = """
**What this is:** **{mod}** features with highest **latent shift** importance—those whose perturbation **moves the model’s latent state** most.

**How to read it:** **Longer bar** = larger shift score within this **top‑N** list (compare lengths across features).

**Takeaway:** Mechanistic “if we nudge this input, the embedding changes a lot.”
"""

_HELP_ATT = """
**What this is:** **{mod}** features with highest **attention** importance from rollout—what the **transformer emphasises** when processing cells.

**How to read it:** **Longer bar** = more average attention mass on that feature (within this top‑N list).

**Takeaway:** Describes **model behaviour** (what it “looks at”), which can differ from causal shift effects.
"""
st.caption("Latent-shift probes, attention rollout, and combined rankings across RNA, ATAC, and Flux.")

df = io.load_df_features()

if df is None:
    st.error(
        "Feature data are not loaded. Ask the maintainer to publish results for this app, then reload."
    )
    st.stop()

st.subheader("Modality spotlight")
st.caption(
    "**Modality spotlight:** three columns (**RNA**, **ATAC**, **Flux**). Each column only shows features "
    "from that modality so you can compare shift impact, attention, and joint ranking **within** RNA, ATAC, or flux."
)
top_n_rank = st.slider("Top N per chart", 10, 55, 20, key="t2_topn")
st.markdown("##### Joint top markers (by mean rank)")
st.caption(
    "The **strongest combined** markers by mean rank (lower mean rank = higher joint shift + attention priority). "
    "Shift and attention bars are **min-max scaled within this top-N list** (0 to 1) so you can compare them on one axis. "
    "Hover a bar for the full feature name."
)
r1a, r1b, r1c = st.columns(3)
for col, mod in zip((r1a, r1b, r1c), ("RNA", "ATAC", "Flux")):
    sm = df[df["modality"] == mod]
    if sm.empty:
        continue
    with col:
        _, _hp = st.columns([1, 0.28])
        with _hp:
            ui.plot_help_popover(_HELP_JOINT.format(mod=mod), key=f"t2_joint_{mod}")
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
        _, _hp = st.columns([1, 0.28])
        with _hp:
            ui.plot_help_popover(_HELP_SHIFT.format(mod=mod), key=f"t2_shift_{mod}")
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
        _, _hp = st.columns([1, 0.28])
        with _hp:
            ui.plot_help_popover(_HELP_ATT.format(mod=mod), key=f"t2_att_{mod}")
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
