"""Flux Analysis: ranked reaction table and download."""

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
**What this is:** The **full FateFormer flux reaction table** for this deployment: one **row** per **reaction** in the metabolic layer, with **joint ranking** and cohort flux statistics from the precomputed results.

**Ranking:** **mean_rank** = combined **shift + attention** priority (**lower** = stronger overall). **rank_shift_in_modal** / **rank_att_in_modal** and **combined_order_mod** are **within-modality** (Flux-only) ranks; **rank_shift** / **rank_att** are **global** across all features. **importance_shift** / **importance_att** are the underlying scores. **top_10_pct** (if present) flags global top-decile membership from the publish step.

**Flux / cohort columns:** **mean_de** / **mean_re** = **mean inferred flux** in **dead-end** vs **reprogramming** samples. **log_fc** = **log₂** fold change between those cohorts for that reaction. **pval_adj** = **adjusted p-value** for that contrast. **group** summarises direction or contrast label when present.

**Context:** **pathway** and **module** annotate the reaction in the reconstruction.

**Use:** Narrow rows with the **substring** and **pathway** controls; use the table’s own **sort** if your Streamlit build exposes it. **Download** saves the **filtered** view as CSV.
"""

st.title("Flux Analysis")
st.caption(
    "**Flux Analysis** ties inferred **reaction flux** to **pathways**, **fate contrasts**, **rankings**, and **model** metadata. "
    "For multimodal **shift**/**attention** summaries, open **Feature Insights**."
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
st.caption(
    "Here is the searchable flux reaction table: every reaction’s FateFormer ranks, cohort flux summaries, and pathway "
    "context, with filters and CSV download."
)
if not _data_ok:
    st.error(_data_msg)
else:
    try:
        _rr_l, _rr_r = st.columns([0.94, 0.06], gap="small", vertical_alignment="center")
    except TypeError:
        _rr_l, _rr_r = st.columns([0.94, 0.06], gap="small")
    with _rr_r:
        ui.plot_help_popover(_HELP_REACTION_TABLE, key="flux_rank_table_help")
    q = st.text_input(
        "Substring filter (reaction name)",
        "",
        key="flux_q",
        help="Keep rows whose **reaction** string contains this text (case-insensitive). Leave empty for no name filter.",
    )
    pw_f = st.multiselect(
        "Pathway",
        sorted(flux["pathway"].dropna().unique().astype(str)),
        default=[],
        key="flux_pw_f",
        help="Keep rows in any of the selected **pathways**. Leave empty to include all pathways.",
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
        help="CSV of the **current filtered** table (same columns as on screen), sorted by **mean_rank**.",
    )
