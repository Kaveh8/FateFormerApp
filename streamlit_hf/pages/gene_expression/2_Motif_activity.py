"""Gene expression: ATAC TF motif deviation and activity."""

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

_HELP_MOTIF_ACTIVITY = """
**Overview:** **ATAC** **TF motif** plots: **differential** activity between fate labels (left), then **per-fate mean** z-scored activity (right). Scores summarize **motif-level** signal from the accessibility layer.

**Left (volcano):** **X** = **mean difference** in motif activity (**reprogramming − dead-end**). **Y** = **−log₁₀ adjusted p** (or a precomputed log-*p* column when the table provides it). **Colour** = **mean rank** (joint FateFormer rank; **lower** = stronger). **Hover** for motif name, *p*, **mean rank**, and cohort fields when present.

**Right (scatter):** **X** / **Y** = **mean z-scored** motif activity in **dead-end** vs **reprogramming** cells. The **y = x** line would mark equal average activity; **above** the diagonal means **higher in reprogramming**. **Colour** = **−log₁₀ adjusted p** (red scale; **higher** = more significant). **Hover** for motif, **mean rank**, and **group**.
"""

st.title("Gene Expression & TF Activity")
st.caption(
    "**Pathways** (Reactome / KEGG) and pathway–gene views; **ATAC motif** deviation and **TF activity** by fate; "
    "**gene** and **motif** tables."
)

df = io.load_df_features()
if df is None:
    st.error("Feature data could not be loaded. Reload after results are published, or contact the maintainer.")
    st.stop()

rna = df[df["modality"] == "RNA"].copy()
atac = df[df["modality"] == "ATAC"].copy()
if rna.empty and atac.empty:
    st.warning("No RNA gene or ATAC motif features are available in the current results.")
    st.stop()

try:
    _ma_h_l, _ma_h_r = st.columns([0.94, 0.06], gap="small", vertical_alignment="center")
except TypeError:
    _ma_h_l, _ma_h_r = st.columns([0.94, 0.06], gap="small")
with _ma_h_l:
    st.subheader("Motif activity")
with _ma_h_r:
    ui.plot_help_popover(_HELP_MOTIF_ACTIVITY, key="ge_motif_page_help")
st.caption(
    "Here, we summarize ATAC TF motif behaviour: differential shift between dead-end and reprogramming (volcano), then "
    "per-fate mean z-scored activity in a scatter."
)

if atac.empty:
    st.warning("No motif-level ATAC features are available in the current results.")
else:
    a1, a2 = st.columns(2, gap="medium")
    with a1:
        st.plotly_chart(plots.motif_chromvar_volcano(atac), width="stretch")
    with a2:
        st.plotly_chart(
            plots.notebook_style_activity_scatter(
                atac,
                title="TF activity (z-score) by fate",
                x_title="Dead-end (TF activity)",
                y_title="Reprogramming (TF activity)",
            ),
            width="stretch",
        )
