"""Gene expression — TF motif activity (chromVAR-style)."""

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

_HELP_MOTIF_VOLC = """
**What this is:** A **volcano‑style** summary of **TF motif** differences from the **ATAC** layer (**chromVAR‑like** scores): **X** = change between fate groups (typically **reprogramming − dead‑end**); **Y** = **significance**.

**How to read it:** **Extreme horizontal** motifs differ most between fates; **higher vertical** motifs are more statistically supported. **Hover** for motif names.

**Takeaway:** Links **chromatin accessibility** motifs to **fate bias** beyond gene‑level RNA.
"""

_HELP_MOTIF_SCATTER = """
**What this is:** **Mean TF motif activity** (**z‑scored**) in **dead‑end** (**X**) versus **reprogramming** (**Y**) cells.

**How to read it:** Points **above the diagonal** are more active in reprogramming; **below** favour dead‑end. **Colour / size** follow the same convention as **Feature Insights** motif views—use **hover** for identifiers.

**Takeaway:** A **direct fate‑vs‑fate** comparison of **regulatory** programmes inferred from accessibility.
"""

st.title("Gene Expression & TF Activity")
st.caption(
    "Pathway enrichment (Reactome / KEGG) and a pathway-gene map; chromVAR-style motif deviations and activity by "
    "fate; sortable gene and motif tables. Use **Feature Insights** for global shift and attention rankings across modalities."
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

st.subheader("Motif activity")
if atac.empty:
    st.warning("No motif-level ATAC features are available in the current results.")
else:
    st.caption(
        "Left: mean motif score difference (reprogramming − dead-end) versus significance. "
        "Right: mean activity in each fate; colour and size follow the same encoding as in **Feature Insights**."
    )
    a1, a2 = st.columns(2, gap="medium")
    with a1:
        _, _hp = st.columns([1, 0.22])
        with _hp:
            ui.plot_help_popover(_HELP_MOTIF_VOLC, key="ge_motif_vol_help")
        st.plotly_chart(plots.motif_chromvar_volcano(atac), width="stretch")
    with a2:
        _, _hp = st.columns([1, 0.22])
        with _hp:
            ui.plot_help_popover(_HELP_MOTIF_SCATTER, key="ge_motif_sc_help")
        st.plotly_chart(
            plots.notebook_style_activity_scatter(
                atac,
                title="TF activity (z-score) by fate",
                x_title="Dead-end (TF activity)",
                y_title="Reprogramming (TF activity)",
            ),
            width="stretch",
        )
