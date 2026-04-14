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

_CELLTAG_MULTI_ARTICLE_URL = "https://www.nature.com/articles/s41587-023-01931-4"

_UMAP_EXPLORER_TITLE = "Validation latent space (UMAP)"
_UMAP_EXPLORER_SUBTITLE = "Hover points for details · drag on the plot to select cells"

_UMAP_EXPLORER_HELP = f"""
**What this is:** The same **2‑D UMAP** as on **Home**: validation **single cells** in **FateFormer**’s **latent space** (**context vector token representation**), summarised across **5-fold cross-validation** (**2,110** cells before filters). Here you **choose what to colour** and **filter** the cloud.

**How to read it:** Each point is one cell. **Colour** comes from **Colour by**: e.g. [**CellTag-Multi**]({_CELLTAG_MULTI_ARTICLE_URL}) **label**, **predicted fate**, **prediction correct / wrong**, **CV fold**, **batch**, which **modalities** are present, or **dominant fate %**. **Axes are unitless** (UMAP preserves *local* neighbourhoods only). **Hover** a point for per-cell fields.

**Using this page:** Use **Filters** to keep modality combinations, restrict **prediction outcome** (all / correct only / wrong only), choose **CV folds**, and set a **dominant fate %** range. In the plot **toolbar** (top right), pick **Box select** or **Lasso select**, then **drag** on the canvas; the app **reruns** and the **Selected points** table fills with those rows. To inspect **one** cell without a selection, scroll to **Inspect by dataset index**.
"""

st.title("Single-Cell Explorer")
st.caption(
    "This page is an interactive **validation UMAP** in FateFormer latent space: you choose how points are **coloured**, "
    "apply **filters**, and can **select** cells on the plot to inspect them in a table or by index."
)

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
    try:
        _sc_umap_plot_col, _sc_umap_help_col = st.columns([0.94, 0.06], gap="small", vertical_alignment="top")
    except TypeError:
        _sc_umap_plot_col, _sc_umap_help_col = st.columns([0.94, 0.06], gap="small")
    with _sc_umap_plot_col:
        fig = plots.latent_scatter(
            plot_df,
            color_opt,
            title=_UMAP_EXPLORER_TITLE,
            width=900,
            height=560,
            marker_size=5.8,
            marker_opacity=0.74,
            subtitle=_UMAP_EXPLORER_SUBTITLE,
        )
        fig.update_layout(margin=dict(l=20, r=12, t=92, b=20), title_font_size=15)
        st.plotly_chart(
            fig,
            width="stretch",
            on_select="rerun",
            key="latent_pick",
            config={"displayModeBar": True, "displaylogo": False},
        )
    with _sc_umap_help_col:
        ui.plot_help_popover(_UMAP_EXPLORER_HELP, key="sc_umap_help")

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
_didx_min = int(df["dataset_idx"].min())
_didx_max = int(df["dataset_idx"].max())
pick = st.number_input(
    "Dataset index",
    min_value=_didx_min,
    max_value=_didx_max,
    value=int(df["dataset_idx"].iloc[0]),
    help=(
        f"The table below is a one-cell summary for the validation set: choose an index from {_didx_min} to {_didx_max} "
        "to see fate labels, model prediction, available modalities, and related fields for that cell."
    ),
)
row = df[df["dataset_idx"] == pick]
if not row.empty:
    st.dataframe(
        formatters.latent_inspector_key_value(row.iloc[0]),
        width="stretch",
        hide_index=True,
    )
