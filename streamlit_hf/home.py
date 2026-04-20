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
_EXPERIMENT_SVG = Path(__file__).resolve().parent / "static" / "experiment.svg"
# Display width (px) for the home-page schematic; SVG scales cleanly at fixed width.
_EXPERIMENT_FIGURE_WIDTH_PX = 380

_CELLTAG_MULTI_ARTICLE_URL = "https://www.nature.com/articles/s41587-023-01931-4"

_APP_NAME = "FateFormer Explorer"
_HERO_EMOJI = "\U0001f9ec"  # DNA (matches HF Space card tone)
_HOME_PIE_TOP_N = 100
_HOME_RANK_TOP_N = 15

_VALIDATION_ROC_AUC = 0.93

_UMAP_HOME_TITLE = "Validation latent space (UMAP)"
_UMAP_HOME_SUBTITLE = "Each point is a cell · colours = experimental fate labels · validation split"
_UMAP_HOME_SUBTITLE_RANK_MISSING = "Feature ranking cache unavailable · UMAP only"

_UMAP_HELP_MD = f"""
**What this is:** A 2‑D **UMAP** of validation **single cells** in the model’s **latent space** (**context vector token representation**), summarised across **5-fold cross-validation**. **2,110** cells are shown.

**How to read it:** Each point is one cell. **Colour** is **experimental fate** from [**CellTag-Multi**]({_CELLTAG_MULTI_ARTICLE_URL}) clonal labels. **Axes are unitless**: UMAP preserves *local* neighbourhoods, not real physical distances, so **nearby points** tend to have similar characteristics in this representation. **Hover** a point for cell-level details. For more detail (interactive UMAP, filters, and metadata), open **Single-Cell Explorer** using the link below.
"""

_GLOBAL_RANK_HELP_MD = """
**What this is:** The **top important fate-predictor markers** for **FateFormer** across its **three modalities** (**RNA** genes, **TF motifs** from chromatin (ATAC), and **flux** reactions), shown as three linked summaries.

**Panels:** **Left / middle** = top features by **latent shift** importance and by **attention** (bars are **min‑max scaled within that panel** so the longest bar is 1). **Right** = **modality mix** (RNA vs ATAC vs Flux) among a pool of **strongest** features by **mean rank** (lower mean rank = higher joint priority).

**How to read it:** Longer bars mean stronger measured influence for that metric. **Colours** mark **modality**. The donut answers: “Among the most important features in this pool, which data type dominates?”.
"""

_APP_SUBTITLE = (
    "A multimodal transformer-based model that jointly encodes RNA, chromatin accessibility, and metabolic flux "
    "to predict single-cell fate, with interpretable attention and latent-shift rankings across omics layers."
)

_EXPERIMENTAL_SYSTEM_MD = f"""
Mouse embryonic fibroblasts (**MEFs**) were reprogrammed toward induced endoderm progenitors (**iEPs**) **in vitro** through *Foxa1* and *HNF4A* induction.

This process produces **mixed outcomes**: some cells successfully reach the **iEP fate**, whereas others diverge into **off-target** trajectories and stall in **dead-end states**.

Using [**CellTag-Multi**]({_CELLTAG_MULTI_ARTICLE_URL}) clonal barcoding, **early cells** could be linked to their **later fate**, which made it possible to ask a central biological question: which programs in **early-state cells**, coordinated **across transcriptional, chromatin, and metabolic layers**, drive successful reprogramming, which ones push cells toward off-target states, and which of these mechanisms could be targeted to improve reprogramming efficiency?
"""

_BIOLOGY_CONTEXT_MARKDOWN = f"""
**How FateFormer addresses this**
- **Multimodal view:** FateFormer integrates **scRNA-seq**, **scATAC-seq**, and **genome-scale metabolic flux** to capture regulatory and metabolic signals that are missed by RNA-only analysis.
- **Grounded in lineage tracing:** The model is trained on a **sparse-modality atlas of more than 150,000 cells**, including **2,110** early cells linked to later outcomes through **CellTag-Multi** clonal barcoding.
- **Biological insight:** FateFormer learns representations across modalities, handles **missing inputs** and **limited labels**, and using **explainability methods** highlights the transcriptional, chromatin, and metabolic programs associated with reprogramming success or off target failure.
"""


def _cache_ok() -> bool:
    return (_CACHE / "latent_umap.pkl").is_file() and (_CACHE / "df_features.parquet").is_file()


def _downsample_latent_df(df, max_points: int = 6500, seed: int = 42):
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed)


def _render_experiment_schematic(width_px: int) -> None:
    """Show the schematic as inline SVG so each group can use CSS hover and native tooltips."""
    raw = _EXPERIMENT_SVG.read_text(encoding="utf-8")
    if raw.lstrip().startswith("<?xml"):
        raw = raw.split("?>", 1)[1].lstrip()
    html_doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><style>
html, body {{
  margin: 0;
  padding: 0;
  background: transparent;
  overflow: visible;
  box-sizing: border-box;
}}
.ff-experiment-svg-wrap {{ width: {width_px}px; max-width: 100%; overflow: visible; }}
.ff-experiment-svg-wrap svg {{ width: 100%; height: auto; display: block; }}
.ff-experiment-svg-wrap svg g[id] {{
  cursor: help;
  transition: filter 0.15s ease;
}}
.ff-experiment-svg-wrap svg g[id]:hover {{
  filter: brightness(1.12) drop-shadow(0 0 1.5px rgba(79, 70, 229, 0.55));
}}
/* Microscope: decorative only (no tooltip); ignore pointer so it does not steal hovers. */
.ff-experiment-svg-wrap svg #microscope,
.ff-experiment-svg-wrap svg #microscope * {{
  pointer-events: none;
}}
.ff-experiment-svg-wrap svg text {{
  cursor: help;
  transition: filter 0.15s ease;
}}
.ff-experiment-svg-wrap svg text:hover {{
  filter: brightness(1.08);
}}
#ff-svgtip {{
  position: fixed;
  left: 0;
  top: 0;
  z-index: 2147483647;
  display: none;
  max-width: min(22rem, calc(100vw - 20px));
  padding: 10px 14px;
  font-size: 15px;
  line-height: 1.5;
  font-family: system-ui, -apple-system, Segoe UI, sans-serif;
  color: #f1f5f9;
  background: #0f172a;
  border-radius: 8px;
  box-shadow: 0 4px 18px rgba(0,0,0,.25);
  pointer-events: none;
}}
</style></head><body>
<div class="ff-experiment-svg-wrap">
{raw}
</div>
<script>
(function () {{
  const tip = document.createElement("div");
  tip.id = "ff-svgtip";
  document.body.appendChild(tip);
  const OFFSET = 14;
  const PAD = 10;
  function placeTip(e) {{
    if (tip.style.display !== "block") return;
    tip.style.visibility = "hidden";
    tip.style.left = "0";
    tip.style.top = "0";
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    tip.style.visibility = "visible";
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const O = OFFSET;
    const P = PAD;
    let x = e.clientX + O;
    let y = e.clientY + O;
    if (y + h + P > vh) y = e.clientY - h - O;
    if (x + w + P > vw) x = e.clientX - w - O;
    if (x + w + P > vw) x = Math.max(P, vw - w - P);
    if (y + h + P > vh) y = Math.max(P, vh - h - P);
    if (x < P) x = P;
    if (y < P) y = P;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }}
  function bind(el) {{
    if (el.closest && el.closest("#microscope")) return;
    const t = el.querySelector(":scope > title");
    if (!t || !t.textContent.trim()) return;
    const txt = t.textContent.trim();
    t.remove();
    el.addEventListener("mouseenter", function (e) {{
      tip.textContent = txt;
      tip.style.display = "block";
      requestAnimationFrame(function () {{ placeTip(e); }});
    }});
    el.addEventListener("mousemove", function (e) {{ placeTip(e); }});
    el.addEventListener("mouseleave", function () {{
      tip.style.display = "none";
    }});
  }}
  document.querySelectorAll(".ff-experiment-svg-wrap svg g[id]").forEach(bind);
  document.querySelectorAll(".ff-experiment-svg-wrap svg text").forEach(bind);
}})();
</script>
</body></html>"""
    st.iframe(html_doc, width=width_px, height="content")


ui.inject_app_styles()
ui.inject_home_landing_styles()

# Bordered Streamlit blocks use overflow that clips iframe tooltips; allow paint past the card edge.
st.markdown(
    """
<style>
section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(iframe) {
  overflow: visible !important;
}
section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(iframe) > div {
  overflow: visible !important;
}
</style>
""",
    unsafe_allow_html=True,
)

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

with st.container(border=True):
    # Wider text column → fewer wrapped lines; tighter gap; center figure vs text when heights differ.
    try:
        fig_col, text_col = st.columns([0.33, 0.67], gap="medium", vertical_alignment="center")
    except TypeError:
        fig_col, text_col = st.columns([0.33, 0.67], gap="medium")
    with fig_col:
        if _EXPERIMENT_SVG.is_file():
            _render_experiment_schematic(_EXPERIMENT_FIGURE_WIDTH_PX)
        else:
            st.caption("Experimental schematic (`static/experiment.svg`) is missing.")
    with text_col:
        st.markdown(_EXPERIMENTAL_SYSTEM_MD)

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
        st.page_link(
            "pages/feature_insights/1_Global_overview.py",
            label="Feature Insights",
            icon=":material/analytics:",
        )
        st.caption("Shift probes, attention rollout, cohort views, and full multimodal tables.")
with c3:
    st.markdown(_NAV_SLOT.format(3), unsafe_allow_html=True)
    with st.container(border=True):
        st.page_link("pages/flux_analysis/5_Interactive_map.py", label="Flux Analysis", icon=":material/account_tree:")
        st.caption("Reaction pathways, differential flux, rankings, and model metadata.")
with c4:
    st.markdown(_NAV_SLOT.format(4), unsafe_allow_html=True)
    with st.container(border=True):
        st.page_link(
            "pages/gene_expression/1_Pathway_enrichment.py",
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
        try:
            _umap_plot_col, _umap_help_col = st.columns([0.94, 0.06], gap="small", vertical_alignment="top")
        except TypeError:
            _umap_plot_col, _umap_help_col = st.columns([0.94, 0.06], gap="small")
        with _umap_plot_col:
            fig_u = plots.latent_scatter(
                plot_umap,
                "label",
                title=_UMAP_HOME_TITLE,
                width=780,
                height=440,
                marker_size=5.2,
                marker_opacity=0.72,
                subtitle=_UMAP_HOME_SUBTITLE,
            )
            fig_u.update_layout(margin=dict(l=20, r=8, t=92, b=20), title_font_size=15)
            st.plotly_chart(
                fig_u,
                width="stretch",
                config={"displayModeBar": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            )
        with _umap_help_col:
            ui.plot_help_popover(
                _UMAP_HELP_MD,
                key="home_umap_help",
                page_link=("pages/1_Single_Cell_Explorer.py", "Single-Cell Explorer"),
            )

    ui.plot_caption_with_help(
        "Global shift and attention · top features (min-max scaled within each bar chart) · modality mix donut (top by mean rank).",
        _GLOBAL_RANK_HELP_MD,
        key="home_global_rank_help",
    )
    fig_g = plots.global_rank_triple_panel(
        df_features,
        top_n=_HOME_RANK_TOP_N,
        top_n_pie=_HOME_PIE_TOP_N,
        chart_outline=False,
        modality_mix_hole=0.66,
        modality_mix_hover_feature_list=True,
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
        try:
            _umap_plot_col2, _umap_help_col2 = st.columns([0.94, 0.06], gap="small", vertical_alignment="top")
        except TypeError:
            _umap_plot_col2, _umap_help_col2 = st.columns([0.94, 0.06], gap="small")
        with _umap_plot_col2:
            fig_u = plots.latent_scatter(
                plot_umap,
                "label",
                title=_UMAP_HOME_TITLE,
                width=820,
                height=480,
                marker_size=5.5,
                marker_opacity=0.72,
                subtitle=_UMAP_HOME_SUBTITLE_RANK_MISSING,
            )
            fig_u.update_layout(margin=dict(l=24, r=12, t=92, b=24), title_font_size=15)
            st.plotly_chart(fig_u, width="stretch", config={"displayModeBar": True, "displaylogo": False})
        with _umap_help_col2:
            ui.plot_help_popover(
                _UMAP_HELP_MD,
                key="home_umap_only_help",
                page_link=("pages/1_Single_Cell_Explorer.py", "Single-Cell Explorer"),
            )
elif df_features is not None:
    ui.plot_caption_with_help(
        "Feature ranking overview · latent UMAP unavailable",
        _GLOBAL_RANK_HELP_MD,
        key="home_global_only_help",
    )
    fig_g = plots.global_rank_triple_panel(
        df_features,
        top_n=_HOME_RANK_TOP_N,
        top_n_pie=_HOME_PIE_TOP_N,
        chart_outline=False,
        modality_mix_hole=0.66,
        modality_mix_hover_feature_list=True,
    )
    fig_g.update_layout(title_text="", margin=dict(l=36, r=36, t=48, b=100))
    st.plotly_chart(fig_g, width="stretch", config={"displayModeBar": False, "displaylogo": False})
else:
    st.info("Charts will appear here once latent and feature caches are available.")
