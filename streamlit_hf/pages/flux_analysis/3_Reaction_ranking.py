"""Flux Analysis — ranked reaction table and download."""

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

_HELP_REACTION_TABLE = """
**What this is:** A **sortable, filterable** version of the **flux reaction** interpretability table (same reactions as elsewhere in Flux Analysis).

**Columns:** Typically include **mean_rank** (overall priority), **shift** / **attention** ranks and scores, **pathway** / **module**, and **differential statistics** (e.g. Log₂FC, adjusted *p*) where computed.

**How to use:** **Filter** by name substring or **pathway**, then **download CSV** for plotting or supplementary material.
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
else:
    flux = df[df["modality"] == "Flux"].copy()
    if flux.empty:
        _data_ok = False
        _data_msg = "There are no flux reactions in the current results."
        flux = None

st.subheader("Reaction ranking")
if not _data_ok:
    st.error(_data_msg)
else:
    ui.plot_caption_with_help(
        "Filter by reaction name or pathway, then inspect or download the ranked flux table.",
        _HELP_REACTION_TABLE,
        key="flux_rank_table_help",
    )
    q = st.text_input("Substring filter (reaction name)", "", key="flux_q")
    pw_f = st.multiselect(
        "Pathway",
        sorted(flux["pathway"].dropna().unique().astype(str)),
        default=[],
        key="flux_pw_f",
    )
    show = flux
    if q.strip():
        show = show[show["feature"].astype(str).str.contains(q, case=False, na=False)]
    if pw_f:
        show = show[show["pathway"].astype(str).isin(pw_f)]
    cols = [
        c
        for c in [
            "mean_rank",
            "feature",
            "rank_shift_in_modal",
            "rank_att_in_modal",
            "combined_order_mod",
            "rank_shift",
            "rank_att",
            "importance_shift",
            "importance_att",
            "top_10_pct",
            "mean_de",
            "mean_re",
            "group",
            "log_fc",
            "pval_adj",
            "pathway",
            "module",
        ]
        if c in show.columns
    ]
    st.dataframe(show[cols].sort_values("mean_rank"), width="stretch", hide_index=True)
    st.download_button(
        "Download Flux table (CSV)",
        show[cols].sort_values("mean_rank").to_csv(index=False).encode("utf-8"),
        file_name="fateformer_flux_filtered.csv",
        mime="text/csv",
        key="flux_dl",
    )
