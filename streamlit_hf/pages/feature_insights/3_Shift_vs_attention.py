"""Feature Insights: shift vs attention rank scatter by modality."""

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

# Native Streamlit tooltips (caption help); plain text reads well in the small ? popover.
_CORR_TABLE_HELP = (
    "Per-modality correlation between attention rank and latent-shift rank across features in that modality "
    "(same features as in the scatters below). Pearson r and Spearman rho measure rank agreement, with p-values. "
    "# features is how many features in that modality were used for the correlation (one rank pair per feature). "
    "Higher |r| means stronger agreement in how features are ordered: a feature that ranks high on shift (small rank; 1 = strongest) "
    "tends to sit in a similar place on attention rank, and the same for weaker features, across that modality."
)

_SCATTER_HELP = (
    "Each dot is one feature in that column: a gene (RNA), TF motif (ATAC), or reaction (Flux). "
    "X = attention rank (1 = strongest in that modality); Y = latent shift rank (1 = strongest). "
    "Ranks on both axes show agreement between methods: near the diagonal means similar ranking; "
    "the dashed trend line is a least-squares fit. Correlation for each modality is in the table above; "
    "stronger r means closer alignment of shift- and attention-based importance as fate predictors. "
    "Point colour is whether that feature sits in the top ~10% by shift rank, attention rank, both, or neither, "
    "using ranks within that modality only (same scale as the axes)."
)

df = io.load_df_features()

if df is None:
    st.error(
        "Feature data are not loaded. Ask the maintainer to publish results for this app, then reload."
    )
    st.stop()

st.title(ui.FEATURE_INSIGHTS_TITLE)
st.caption(ui.FEATURE_INSIGHTS_CAPTION)
st.subheader("Shift vs attention")
st.caption(
    "Here, we explore how much latent-shift and attention-rollout explanations agree on feature importance within each "
    "modality. A correlation table quantifies rank agreement; scatter plots pair each feature’s two ranks "
    "(1 = strongest in that modality)."
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
    st.caption(
        "Rank correlation by modality",
        help=_CORR_TABLE_HELP,
    )
    st.dataframe(pd.DataFrame(corr_rows), hide_index=True, width="stretch")

st.caption(
    "Rank scatter by modality",
    help=_SCATTER_HELP,
)
rc1, rc2, rc3 = st.columns(3)
for col, mod in zip((rc1, rc2, rc3), ("RNA", "ATAC", "Flux")):
    with col:
        sub_m = df[df["modality"] == mod]
        st.plotly_chart(
            plots.rank_scatter_shift_vs_attention(sub_m, mod),
            width="stretch",
        )
