"""Landing page for the FateFormer Explorer Streamlit hub."""

from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np
import streamlit as st

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from streamlit_hf.lib import io
from streamlit_hf.lib import plots
from streamlit_hf.lib import ui

_CACHE = Path(__file__).resolve().parent / "cache"

_APP_NAME = "FateFormer Explorer"
_HERO_EMOJI = "\U0001f9ec"  # DNA (matches HF Space card tone)
_HOME_PIE_TOP_N = 100
_HOME_RANK_TOP_N = 15

_VALIDATION_ROC_AUC = 0.93

_UMAP_HOME_TITLE = "Validation latent space (UMAP)"

_APP_SUBTITLE = (
    "A multimodal transformer-based model that jointly encodes RNA, chromatin accessibility, and metabolic flux "
    "to predict single-cell fate, with interpretable attention and latent-shift rankings across omics layers."
)

_BIOLOGY_CONTEXT_MARKDOWN = """
**At a glance**

- **Biological setting:** **FateFormer** models **direct reprogramming** from mouse embryonic fibroblasts (**MEFs**) to induced endoderm progenitors (**iEPs**), combining **transcriptome (scRNA-seq)**, **chromatin (scATAC-seq)**, and **genome-scale metabolic flux** so fate is not inferred from RNA alone; epigenetic and metabolic context matter.
- **Data & labels:** Trained on a **large sparse-modality** atlas (**>150,000** cells); **2,110** early cells carry **CellTag-Multi** clonal fate tags, the same experimental labels used to colour validation cells in **UMAP** views here.
- **Model design:** A **transformer** learns **shared representations** across modalities, handles **missing modalities** and **scarce fate labels**, and ties early transcription, chromatin accessibility, and metabolic activity to **later lineage outcomes**, going beyond RNA-only views of reprogramming.
"""


def _cache_ok() -> bool:
    return (_CACHE / "latent_umap.pkl").is_file() and (_CACHE / "df_features.parquet").is_file()


def _downsample_latent_df(df, max_points: int = 6500, seed: int = 42):
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed)


ui.inject_app_styles()
ui.inject_home_landing_styles()

st.markdown(
    f"""<div class="ff-hero"><div class="ff-hero-inner"><div class="ff-hero-text">
<div class="ff-hero-title-row">
<span class="ff-hero-emoji" aria-hidden="true">{_HERO_EMOJI}</span>
<h1>{html.escape(_APP_NAME)}</h1>
</div>
<p class="ff-hero-sub">{html.escape(_APP_SUBTITLE)}</p>
</div></div></div>""",
    unsafe_allow_html=True,
)

bundle = io.load_latent_bundle()
df_features = io.load_df_features()
samples = io.load_samples_df()
ready = _cache_ok()

if not ready:
    st.warning(
        "Precomputed validation caches are incomplete or missing. "
        "Publish `latent_umap.pkl` and `df_features.parquet` under `streamlit_hf/cache/`, then reload."
    )

# --- Metrics strip ---
mcols = st.columns(4)
if bundle is not None:
    n_cells = len(bundle["umap_x"])
    with mcols[0]:
        st.metric("Validation cells", f"{n_cells:,}")
    with mcols[1]:
        st.metric("Validation ROC-AUC", f"{_VALIDATION_ROC_AUC:.2f}")
else:
    with mcols[0]:
        st.metric("Validation cells", "n/a")
    with mcols[1]:
        st.metric("Validation ROC-AUC", "n/a")

if df_features is not None:
    nf = len(df_features)
    n_mod = df_features["modality"].nunique() if "modality" in df_features.columns else 0
    with mcols[2]:
        st.metric("Ranked features", f"{nf:,}")
    with mcols[3]:
        st.metric("Modalities", str(n_mod) if n_mod else "n/a")
else:
    with mcols[2]:
        st.metric("Ranked features", "n/a")
    with mcols[3]:
        st.metric("Modalities", "n/a")

# --- Workspace cards (directly under metrics); hidden spans pair with CSS for per-card colours ---
_NAV_SLOT = '<span id="ff-nav-slot-{}" class="ff-nav-slot-marker" aria-hidden="true"></span>'
c1, c2, c3, c4 = st.columns(4, gap="small")
with c1:
    st.markdown(_NAV_SLOT.format(1), unsafe_allow_html=True)
    with st.container(border=True):
        st.page_link("pages/1_Single_Cell_Explorer.py", label="Single-Cell Explorer", icon=":material/scatter_plot:")
        st.caption("UMAP, filters, and per-cell inspection: fate, prediction, fold, batch, modalities.")
with c2:
    st.markdown(_NAV_SLOT.format(2), unsafe_allow_html=True)
    with st.container(border=True):
        st.page_link("pages/2_Feature_insights.py", label="Feature Insights", icon=":material/analytics:")
        st.caption("Shift probes, attention rollout, cohort views, and full multimodal tables.")
with c3:
    st.markdown(_NAV_SLOT.format(3), unsafe_allow_html=True)
    with st.container(border=True):
        st.page_link("pages/3_Flux_analysis.py", label="Flux Analysis", icon=":material/account_tree:")
        st.caption("Reaction pathways, differential flux, rankings, and model metadata.")
with c4:
    st.markdown(_NAV_SLOT.format(4), unsafe_allow_html=True)
    with st.container(border=True):
        st.page_link(
            "pages/4_Gene_expression_analysis.py",
            label="Gene Expression & TF Activity",
            icon=":material/genetics:",
        )
        st.caption("Pathway enrichment, motif activity, and searchable gene tables.")

st.markdown('<p class="ff-section-label">Overview</p>', unsafe_allow_html=True)

# --- Snapshot charts ---
if bundle is not None and df_features is not None:
    latent_df = io.latent_join_samples(bundle, samples)
    plot_umap = _downsample_latent_df(latent_df)
    row1_story, row1_umap = st.columns([0.38, 0.62], gap="large")
    with row1_story:
        st.markdown(_BIOLOGY_CONTEXT_MARKDOWN)
    with row1_umap:
        st.caption("Each point is a cell · colours = experimental fate labels · validation split")
        fig_u = plots.latent_scatter(
            plot_umap,
            "label",
            title=_UMAP_HOME_TITLE,
            width=780,
            height=440,
            marker_size=5.2,
            marker_opacity=0.72,
        )
        fig_u.update_layout(margin=dict(l=20, r=8, t=52, b=20), title_font_size=15)
        st.plotly_chart(
            fig_u,
            width="stretch",
            config={"displayModeBar": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )

    st.caption("Global shift and attention · top features by importance (min-max scaled within each bar chart) · modality mix as donut (top by mean rank).")
    fig_g = plots.global_rank_triple_panel(
        df_features,
        top_n=_HOME_RANK_TOP_N,
        top_n_pie=_HOME_PIE_TOP_N,
        chart_outline=False,
        modality_mix_hole=0.66,
    )
    fig_g.update_layout(title_text="", margin=dict(l=36, r=36, t=48, b=100))
    fig_g.update_annotations(font_size=12)
    st.plotly_chart(
        fig_g,
        width="stretch",
        config={"displayModeBar": False, "displaylogo": False},
    )
elif bundle is not None:
    latent_df = io.latent_join_samples(bundle, samples)
    plot_umap = _downsample_latent_df(latent_df)
    u_story, u_map = st.columns([0.38, 0.62], gap="large")
    with u_story:
        st.markdown(_BIOLOGY_CONTEXT_MARKDOWN)
    with u_map:
        st.caption("Feature ranking cache unavailable · UMAP only")
        fig_u = plots.latent_scatter(
            plot_umap,
            "label",
            title=_UMAP_HOME_TITLE,
            width=820,
            height=480,
            marker_size=5.5,
            marker_opacity=0.72,
        )
        fig_u.update_layout(margin=dict(l=24, r=12, t=52, b=24), title_font_size=15)
        st.plotly_chart(fig_u, width="stretch", config={"displayModeBar": True, "displaylogo": False})
elif df_features is not None:
    st.caption("Feature ranking overview · latent UMAP unavailable")
    fig_g = plots.global_rank_triple_panel(
        df_features,
        top_n=_HOME_RANK_TOP_N,
        top_n_pie=_HOME_PIE_TOP_N,
        chart_outline=False,
        modality_mix_hole=0.66,
    )
    fig_g.update_layout(title_text="", margin=dict(l=36, r=36, t=48, b=100))
    st.plotly_chart(fig_g, width="stretch", config={"displayModeBar": False, "displaylogo": False})
else:
    st.info("Charts will appear here once latent and feature caches are available.")
