"""Metabolic flux: pathway map, differential views, reaction ranking table, metabolic model metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from streamlit_hf.lib import io
from streamlit_hf.lib import plots
from streamlit_hf.lib import ui

ui.inject_app_styles()

st.title("Flux Analysis")
st.caption(
    "Reaction-level flux: how pathways, statistics, and model rankings line up. "
    "For global rank bars and shift vs. attention scatter, open **Feature insights**."
)

df = io.load_df_features()
if df is None:
    st.error(
        "Flux and feature data are not loaded in this session. Reload the app after the maintainer has published "
        "fresh results, or ask them to check the deployment."
    )
    st.stop()

flux = df[df["modality"] == "Flux"].copy()
if flux.empty:
    st.warning("There are no flux reactions in the current results.")
    st.stop()

meta = io.load_metabolic_model_metadata()

tab_map, tab_bio, tab_rank, tab_meta = st.tabs(
    [
        "Pathway map",
        "Differential & fate",
        "Reaction ranking",
        "Metabolic model metadata",
    ]
)

with tab_map:
    st.caption(
        "**Left:** sunburst of the strongest reactions by mean rank, grouped by pathway. **Right:** heatmaps for the "
        "same reactions: pathway, differential Log₂FC, and statistical significance, aligned row by row. "
        "Ranked reaction table: **Reaction Ranking**. Curated model edges: **Metabolic model metadata**."
    )
    try:
        c1, c2 = st.columns([1.05, 0.95], gap="medium", vertical_alignment="top")
    except TypeError:
        c1, c2 = st.columns([1.05, 0.95], gap="medium")
    with c1:
        n_sb = st.slider("Reactions in sunburst", 25, 90, 52, key="flux_sb_n")
        st.plotly_chart(plots.flux_pathway_sunburst(flux, max_features=n_sb), width="stretch")
    with c2:
        top_n_nb = st.slider("Reactions in annotation + profile", 12, 40, 26, key="flux_nb_n")
        st.plotly_chart(
            plots.flux_reaction_annotation_panel(flux, top_n=top_n_nb, metric="mean_rank"),
            width="stretch",
        )
        st.plotly_chart(
            plots.flux_model_metric_profile(flux, top_n=min(top_n_nb, 24), metric="mean_rank"),
            width="stretch",
        )

with tab_bio:
    st.caption(
        "**Volcano:** differential Log₂FC versus significance (−log₁₀ adjusted p); colour shows overall mean rank. "
        "Points with essentially no fold change and a zero adjusted p-value are removed as unreliable. "
        "**Scatter:** average measured flux in dead-end versus reprogramming cells; point size reflects combined shift "
        "and attention strength; colours mark pathway (largest groups shown, others grouped as *Other*)."
    )
    b1, b2 = st.columns(2)
    with b1:
        st.plotly_chart(plots.flux_volcano(flux), width="stretch")
    with b2:
        st.plotly_chart(plots.flux_dead_end_vs_reprogram_scatter(flux), width="stretch")

with tab_rank:
    st.caption("Filter by reaction name or pathway, then inspect or download the ranked flux table.")
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

with tab_meta:
    st.caption(
        "Directed substrate-to-product steps from the reference model, merged with this flux table where reaction names match."
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
