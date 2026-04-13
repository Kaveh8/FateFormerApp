"""Multimodal feature importance: ranks, attention by prediction, tables."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from streamlit_hf.lib import io
from streamlit_hf.lib import plots
from streamlit_hf.lib import ui

ui.inject_app_styles()

st.title("Feature Insights")
st.caption("Latent-shift probes, attention rollout, and combined rankings across RNA, ATAC, and Flux.")

df = io.load_df_features()
att = io.load_attention_summary()

if df is None:
    st.error(
        "Feature data are not loaded. Ask the maintainer to publish results for this app, then reload."
    )
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Global overview",
        "Modality spotlight",
        "Shift vs attention",
        "Attention vs prediction",
        "Full table",
    ]
)

# ----- Tab 1 -----
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        top_n_bars = st.slider(
            "Top N (shift & attention bars)",
            10,
            45,
            20,
            key="t1_topn_bars",
        )
    with c2:
        top_n_pie = st.slider(
            "Pool size (mean-rank pie)",
            50,
            250,
            100,
            key="t1_topn_pie",
        )
    st.plotly_chart(
        plots.global_rank_triple_panel(df, top_n=top_n_bars, top_n_pie=top_n_pie),
        width="stretch",
    )
    st.caption(
        "Bars: **global** top features by shift impact and by mean attention (min-max scaled); "
        "colour = modality. Pie: RNA / ATAC / Flux mix among the lowest mean-rank features in that pool."
    )

# ----- Tab 2: RNA / ATAC / Flux columns -----
with tab2:
    st.caption(
        "**Modality spotlight:** three columns (**RNA**, **ATAC**, **Flux**). Each column only shows features "
        "from that modality so you can compare shift impact, attention, and joint ranking **within** RNA, ATAC, or flux."
    )
    top_n_rank = st.slider("Top N per chart", 10, 55, 20, key="t2_topn")
    st.subheader("Joint top markers (by mean rank)")
    st.caption(
        "The **strongest combined** markers by mean rank (lower mean rank = higher joint shift + attention priority). "
        "Shift and attention bars are **min-max scaled within this top-N list** (0 to 1) so you can compare them on one axis. "
        "Hover a bar for the full feature name."
    )
    r1a, r1b, r1c = st.columns(3)
    for col, mod in zip((r1a, r1b, r1c), ("RNA", "ATAC", "Flux")):
        sm = df[df["modality"] == mod]
        if sm.empty:
            continue
        with col:
            st.plotly_chart(
                plots.joint_shift_attention_top_features(sm, mod, top_n_rank),
                width="stretch",
            )
    st.subheader("Shift importance")
    r2a, r2b, r2c = st.columns(3)
    for col, mod in zip((r2a, r2b, r2c), ("RNA", "ATAC", "Flux")):
        sm = df[df["modality"] == mod]
        if sm.empty:
            continue
        colc = plots.MODALITY_COLOR.get(mod, plots.PALETTE[0])
        sub = sm.nlargest(top_n_rank, "importance_shift").sort_values("importance_shift", ascending=True)
        with col:
            st.plotly_chart(
                plots.rank_bar(
                    sub,
                    "importance_shift",
                    "feature",
                    f"{mod}: shift · top {top_n_rank}",
                    colc,
                    xaxis_title="Latent shift importance",
                ),
                width="stretch",
            )
    st.subheader("Attention importance")
    r3a, r3b, r3c = st.columns(3)
    for col, mod in zip((r3a, r3b, r3c), ("RNA", "ATAC", "Flux")):
        sm = df[df["modality"] == mod]
        if sm.empty:
            continue
        colc = plots.MODALITY_COLOR.get(mod, plots.PALETTE[0])
        sub = sm.nlargest(top_n_rank, "importance_att").sort_values("importance_att", ascending=True)
        with col:
            st.plotly_chart(
                plots.rank_bar(
                    sub,
                    "importance_att",
                    "feature",
                    f"{mod}: attention · top {top_n_rank}",
                    colc,
                    xaxis_title="Attention importance",
                ),
                width="stretch",
            )

# ----- Tab 3 -----
with tab3:
    st.caption(
        "Each point is **one feature** within its modality. **Attention rank** is on the horizontal axis and **shift rank** "
        "on the vertical axis (1 = strongest in that modality for that metric). Features near the diagonal rank similarly "
        "for both; the **red dashed line** is a straight-line trend (least-squares fit) through the cloud."
    )
    corr_rows = []
    for mod in ("RNA", "ATAC", "Flux"):
        sm = df[df["modality"] == mod]
        if sm.empty:
            continue
        cor = plots.modality_shift_attention_rank_stats(sm)
        if cor.get("n", 0) >= 3:
            corr_rows.append(
                {
                    "Modality": mod,
                    "# features": cor["n"],
                    "Pearson r": f"{cor['pearson_r']:.3f}",
                    "Pearson p": f"{cor['pearson_p']:.2e}",
                    "Spearman ρ": f"{cor['spearman_r']:.3f}",
                    "Spearman p": f"{cor['spearman_p']:.2e}",
                }
            )
    if corr_rows:
        st.dataframe(pd.DataFrame(corr_rows), hide_index=True, width="stretch")
    rc1, rc2, rc3 = st.columns(3)
    for col, mod in zip((rc1, rc2, rc3), ("RNA", "ATAC", "Flux")):
        with col:
            sub_m = df[df["modality"] == mod]
            st.plotly_chart(
                plots.rank_scatter_shift_vs_attention(sub_m, mod),
                width="stretch",
            )

# ----- Tab 4 -----
with tab4:
    with st.expander("What is this?", expanded=False):
        st.markdown(
            "Bars show **mean attention weights** (from rollout) averaged over validation cells, split by **what the "
            "model predicted** for each cell: all validation cells together, only cells called **dead-end**, or only "
            "cells called **reprogramming**. This reflects **model behaviour**, not the true fate label."
        )
    cohort_mode = st.selectbox(
        "Cohort view",
        [
            "compare",
            "all",
            "dead_end",
            "reprogramming",
        ],
        format_func=lambda x: {
            "compare": "Compare cohorts (grouped bars)",
            "all": "All validation samples (mean attention)",
            "dead_end": "Mean attention when prediction = dead-end",
            "reprogramming": "Mean attention when prediction = reprogramming",
        }[x],
        key="t4_cohort",
        help=(
            "Choose which validation cells contribute to the average. **All validation samples** uses every validation "
            "cell. The prediction-specific options use only cells where the model output was dead-end or reprogramming, "
            "so you can see which features receive more weight when the model leans each way."
        ),
    )
    top_n_att = st.slider("Top N", 6, 28, 15, key="t4_topn")
    if not att or "fi_att" not in att:
        st.warning(
            "Attention summaries are not available in this session. That view needs a full publish from the maintainer."
        )
    else:
        ac1, ac2, ac3 = st.columns(3)
        for col, mod in zip((ac1, ac2, ac3), ("RNA", "ATAC", "Flux")):
            with col:
                st.plotly_chart(
                    plots.attention_cohort_view(att["fi_att"], mod, top_n=top_n_att, mode=cohort_mode),
                    width="stretch",
                )
        if "rollout_mean" in att and "slices" in att:
            st.subheader("Mean rollout weight")
            if cohort_mode == "compare":
                roll_cohort = st.selectbox(
                    "Rollout table: average over",
                    ["all", "dead_end", "reprogramming"],
                    format_func=lambda x: {
                        "all": "All validation samples",
                        "dead_end": "Cells predicted dead-end",
                        "reprogramming": "Cells predicted reprogramming",
                    }[x],
                    key="t4_roll",
                    help="Pick which validation subset is used for the mean rollout vector in the tables below.",
                )
            else:
                roll_cohort = cohort_mode
                st.caption(
                    "Rollout tables use the **same cohort** as the bar charts above (batch-embedding tokens are omitted)."
                )
            rc1, rc2, rc3 = st.columns(3)
            for col, mod in zip((rc1, rc2, rc3), ("RNA", "ATAC", "Flux")):
                with col:
                    rm = att["rollout_mean"]
                    vec_all = rm.get(roll_cohort)
                    if vec_all is None:
                        vec_all = rm["all"]
                    sl = att["slices"][mod]
                    vec = vec_all[sl["start"] : sl["stop"]]
                    names = att["feature_names"][sl["start"] : sl["stop"]]
                    mini = plots.rollout_top_features_table(names, vec, top_n_att)
                    st.caption(mod)
                    st.dataframe(mini, hide_index=True, width="stretch")

# ----- Tab 5 -----
with tab5:
    scope = st.radio(
        "Table scope",
        ["All modalities", "Single modality"],
        horizontal=True,
        key="t5_scope",
    )
    mod_tbl = "all"
    if scope == "Single modality":
        mod_tbl = st.selectbox("Modality", ["RNA", "ATAC", "Flux"], key="t5_mod")
        tbl = df[df["modality"] == mod_tbl].copy()
    else:
        tbl = df.copy()
    show_cols = [
        c
        for c in [
            "mean_rank",
            "feature",
            "modality",
            "rank_shift_in_modal",
            "rank_att_in_modal",
            "combined_order_mod",
            "rank_shift",
            "rank_att",
            "importance_shift",
            "importance_att",
            "top_10_pct",
            "group",
            "log_fc",
            "pval_adj",
            "pathway",
            "module",
        ]
        if c in tbl.columns
    ]
    st.caption(
        "All rows for the chosen scope, sorted by **mean rank** (lower = stronger joint shift + attention priority). "
        "Use the dataframe search / sort in the table toolbar to narrow down."
    )
    full_view = tbl[show_cols].sort_values("mean_rank")
    st.dataframe(full_view, width="stretch", hide_index=True)
    suffix = mod_tbl if scope == "Single modality" else "all"
    st.download_button(
        "Download table (CSV)",
        full_view.to_csv(index=False).encode("utf-8"),
        file_name=f"fateformer_features_{suffix}.csv",
        mime="text/csv",
        key="t5_dl",
    )
