"""Flux Analysis — metabolic model metadata merged with flux table."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from streamlit_hf.lib import io
from streamlit_hf.lib import ui

ui.inject_app_styles()

_HELP_MODEL_META = """
**What this is:** **Directed edges** from the **genome‑scale metabolic model** (substrate → product reactions), **merged** with this app’s **flux interpretability table** where reaction identifiers match.

**How to read it:** Each row is a **model step** you can relate to **pathways** and **model modules**. Use **Model scope** to zoom to one **supermodule** or view **all** edges.

**Takeaway:** Connects **curated biochemistry** (stoichiometry / wiring) to **data‑driven rankings** from FateFormer.
"""

st.title("Flux Analysis")
st.caption(
    "Reaction-level flux: how pathways, statistics, and model rankings line up. "
    "For global rank bars and shift vs. attention scatter, open **Feature insights**."
)

try:
    df = io.load_df_features()
except Exception:
    df = None

_data_ok = True
if df is None:
    _data_ok = False
    _data_msg = (
        "Flux and feature data are not loaded in this session. Reload the app after the maintainer has published "
        "fresh results, or ask them to check the deployment."
    )
    flux = None
    meta = None
else:
    flux = df[df["modality"] == "Flux"].copy()
    if flux.empty:
        _data_ok = False
        _data_msg = "There are no flux reactions in the current results."
        flux = None
    meta = io.load_metabolic_model_metadata()

st.subheader("Metabolic model metadata")
if not _data_ok:
    st.error(_data_msg)
else:
    ui.plot_caption_with_help(
        "Directed substrate-to-product steps from the reference model, merged with this flux table where reaction names match.",
        _HELP_MODEL_META,
        key="flux_model_meta_help",
    )
    if meta is None or meta.empty:
        st.warning("Metabolic model metadata is not available in this build.")
    else:
        sm_ids = sorted(meta["Supermodule_id"].dropna().unique().astype(int).tolist())
        graph_labels = ["All modules"]
        for sid in sm_ids:
            cls = str(meta.loc[meta["Supermodule_id"] == sid, "Super.Module.class"].iloc[0])
            graph_labels.append(f"{sid}: {cls}")
        tix = st.selectbox(
            "Model scope",
            range(len(graph_labels)),
            format_func=lambda i: graph_labels[i],
            key="flux_model_scope",
            help="Show every step in the model, or restrict to one functional module.",
        )
        supermodule_id = None if tix == 0 else sm_ids[tix - 1]
        tbl = io.build_metabolic_model_table(meta, flux, supermodule_id=supermodule_id)
        st.dataframe(tbl, width="stretch", hide_index=True)
        st.download_button(
            "Download metabolic model metadata (CSV)",
            tbl.to_csv(index=False).encode("utf-8"),
            file_name="fateformer_metabolic_model_edges.csv",
            mime="text/csv",
            key="flux_model_dl",
        )
