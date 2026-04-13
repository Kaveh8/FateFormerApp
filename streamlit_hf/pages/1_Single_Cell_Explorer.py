"""Interactive UMAP of multimodal latent space (validation folds)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from streamlit_hf.lib import formatters
from streamlit_hf.lib import io
from streamlit_hf.lib import plots
from streamlit_hf.lib import ui

ui.inject_app_styles()

st.title("Single-Cell Explorer")
st.caption("Explore validation cells in 2-D UMAP space: colour and filter to compare fates, predictions, and modalities.")

bundle = io.load_latent_bundle()
if bundle is None:
    st.error("Latent maps are not available in this session. Ask the maintainer to publish results, then reload.")
    st.stop()

samples = io.load_samples_df()
df = io.latent_join_samples(bundle, samples)

left, right = st.columns([0.36, 0.64], gap="large")

with left:
    st.markdown('<p class="latent-panel-title">Colour by</p>', unsafe_allow_html=True)
    color_opt = st.selectbox(
        "Hue",
        [
            "label",
            "predicted_class",
            "correct",
            "fold",
            "batch_no",
            "modality_label",
            "pct",
        ],
        format_func=lambda x: {
            "label": "CellTag-Multi label",
            "predicted_class": "Predicted fate",
            "correct": "Prediction correct",
            "fold": "CV fold",
            "batch_no": "Batch",
            "modality_label": "Available modalities",
            "pct": "Dominant fate %",
        }[x],
        label_visibility="collapsed",
        help="Which variable sets the colour of each point on the UMAP.",
    )

    st.markdown('<p class="latent-panel-title latent-panel-title-gap">Filters</p>', unsafe_allow_html=True)
    mod_labels = sorted(df["modality_label"].astype(str).unique())
    mod_pick = st.multiselect(
        "Available modalities",
        mod_labels,
        default=mod_labels,
        help="Keep cells whose modality combination matches your selection (RNA/ATAC measured where present; flux inferred).",
    )
    only_correct = st.selectbox(
        "Prediction outcome",
        ["All", "Correct only", "Wrong only"],
        help="Restrict to cells where the model was correct, incorrect, or show all.",
    )
    folds = sorted(df["fold"].unique())
    fold_pick = st.multiselect(
        "CV folds",
        folds,
        default=folds,
        help="Validation cross-validation folds to include (each fold’s held-out cells).",
    )
    pct_rng = st.slider(
        "Dominant fate % range",
        0.0,
        100.0,
        (0.0, 100.0),
        1.0,
        help="Keep cells whose dominant lineage probability (percent) falls in this range.",
    )

plot_df = df[df["fold"].isin(fold_pick) & df["modality_label"].isin(mod_pick)].copy()
plot_df = plot_df[(plot_df["pct"] >= pct_rng[0]) & (plot_df["pct"] <= pct_rng[1])]
if only_correct == "Correct only":
    plot_df = plot_df[plot_df["correct"]]
elif only_correct == "Wrong only":
    plot_df = plot_df[~plot_df["correct"]]

if plot_df.empty:
    st.warning("No points after filters. Relax the filters and try again.")
    st.stop()

with right:
    fig = plots.latent_scatter(
        plot_df,
        color_opt,
        title="Validation latent space (UMAP)",
        width=900,
        height=560,
        marker_size=5.8,
        marker_opacity=0.74,
    )
    st.plotly_chart(fig, width="stretch", on_select="rerun", key="latent_pick")

st.subheader("Selected points")
state = st.session_state.get("latent_pick")
points = []
if isinstance(state, dict):
    sel = state.get("selection") or {}
    if isinstance(sel, dict):
        points = sel.get("points") or []
if points:
    idxs = [int(p["point_index"]) for p in points if "point_index" in p]
    idxs = [i for i in idxs if 0 <= i < len(plot_df)]
    if idxs:
        sub = plot_df.iloc[idxs]
        disp = formatters.prepare_latent_display_dataframe(sub)
        st.dataframe(
            disp,
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning(
            "A selection was reported but no valid points matched the current filtered view. "
            "Try selecting again after changing filters, or pick a row via **Inspect by dataset index**."
        )
else:
    st.info(
        "This table fills in when you **select points on the UMAP**. "
        "In the chart’s top-right toolbar, choose **Box select** or **Lasso select**, "
        "then drag over the dots; the page reruns and rows for those cells appear here. "
        "To inspect one cell without using the lasso, scroll down to **Inspect by dataset index**."
    )

st.subheader("Inspect by dataset index")
pick = st.number_input(
    "Dataset index",
    min_value=int(df["dataset_idx"].min()),
    max_value=int(df["dataset_idx"].max()),
    value=int(df["dataset_idx"].iloc[0]),
    help="Index `ind` in your sample table; aligns one validation cell to this row.",
)
row = df[df["dataset_idx"] == pick]
if not row.empty:
    st.dataframe(
        formatters.latent_inspector_key_value(row.iloc[0]),
        width="stretch",
        hide_index=True,
    )
