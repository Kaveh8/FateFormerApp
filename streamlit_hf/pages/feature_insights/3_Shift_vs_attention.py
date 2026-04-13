"""Feature Insights — shift vs attention rank scatter by modality."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from streamlit_hf.lib import io
from streamlit_hf.lib import plots
from streamlit_hf.lib import ui

ui.inject_app_styles()

_HELP_SHIFT_VS_ATT = """
**What this is:** Each **dot** is **one {mod} feature**. **X** = rank by **attention** (1 = strongest in this modality); **Y** = rank by **latent shift** (1 = strongest).

**How to read it:** Points **on the diagonal** rank similarly for both metrics. The **red dashed line** is a **least‑squares trend**—it summarises whether higher attention rank tends to pair with higher shift rank in this modality.

**Takeaway:** Features **far from the trend** are interesting: strong in one lens but not the other (e.g. high attention, lower shift, or the reverse).
"""

st.title("Feature Insights")
st.caption("Latent-shift probes, attention rollout, and combined rankings across RNA, ATAC, and Flux.")

df = io.load_df_features()

if df is None:
    st.error(
        "Feature data are not loaded. Ask the maintainer to publish results for this app, then reload."
    )
    st.stop()

st.subheader("Shift vs attention")
st.caption(
    "Each point is **one feature** within its modality. **Attention rank** is on the horizontal axis and **shift rank** "
    "on the vertical axis (1 = strongest in that modality for that metric). Features near the diagonal rank similarly "
    "for both; the **red dashed line** is a straight-line trend (least-squares fit) through the cloud."
)
corr_rows = []
for mod in ("RNA", "ATAC", "Flux"):
    sm = df[df["modality"] == mod]
    if sm.empty:
        continue
    cor = plots.modality_shift_attention_rank_stats(sm)
    if cor.get("n", 0) >= 3:
        corr_rows.append(
            {
                "Modality": mod,
                "# features": cor["n"],
                "Pearson r": f"{cor['pearson_r']:.3f}",
                "Pearson p": f"{cor['pearson_p']:.2e}",
                "Spearman ρ": f"{cor['spearman_r']:.3f}",
                "Spearman p": f"{cor['spearman_p']:.2e}",
            }
        )
if corr_rows:
    st.dataframe(pd.DataFrame(corr_rows), hide_index=True, width="stretch")
rc1, rc2, rc3 = st.columns(3)
for col, mod in zip((rc1, rc2, rc3), ("RNA", "ATAC", "Flux")):
    with col:
        sub_m = df[df["modality"] == mod]
        _, _hp = st.columns([1, 0.28])
        with _hp:
            ui.plot_help_popover(_HELP_SHIFT_VS_ATT.format(mod=mod), key=f"t3_scatter_{mod}")
        st.plotly_chart(
            plots.rank_scatter_shift_vs_attention(sub_m, mod),
            width="stretch",
        )
