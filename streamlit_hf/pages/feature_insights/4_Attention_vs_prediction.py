"""Feature Insights — attention by predicted cohort."""

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

_HELP_ATT_COHORT_BARS = """
**What this is:** **Mean attention** (rollout) on each **feature token**, averaged over validation cells and split by **what the model predicted** for those cells.

**Cohort menu:** **Compare** shows cohorts **side‑by‑side**. **All / dead‑end / reprogramming** restrict the average to that predicted class only.

**Important:** Uses **predicted** fate, **not** the experimental label—this is **model behaviour**, useful for comparing what the network emphasises when it leans each way.

**How to read:** **Longer bar** = more cumulative attention on that feature (among the **top‑N** shown). **Hover** for numeric detail.
"""

_HELP_ROLLOUT_TABLE = """
**What this is:** The same **mean rollout vector** as the bars, but as a **sortable table** of the strongest **{mod}** tokens.

**How to read:** Rows are **ranked** by weight in the selected cohort. **Batch** embedding tokens are omitted from this view.

**Takeaway:** Lets you **copy names** or scan exact ordering beyond the bar chart.
"""

st.title("Feature Insights")
st.caption("Latent-shift probes, attention rollout, and combined rankings across RNA, ATAC, and Flux.")

df = io.load_df_features()
att = io.load_attention_summary()

if df is None:
    st.error(
        "Feature data are not loaded. Ask the maintainer to publish results for this app, then reload."
    )
    st.stop()

st.subheader("Attention vs prediction")
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
            _, _hp = st.columns([1, 0.28])
            with _hp:
                ui.plot_help_popover(_HELP_ATT_COHORT_BARS, key=f"t4_bar_{mod}_{cohort_mode}")
            st.plotly_chart(
                plots.attention_cohort_view(att["fi_att"], mod, top_n=top_n_att, mode=cohort_mode),
                width="stretch",
            )
    if "rollout_mean" in att and "slices" in att:
        st.markdown("##### Mean rollout weight")
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
                cap1, cap2 = st.columns([0.82, 0.18])
                with cap1:
                    st.caption(mod)
                with cap2:
                    ui.plot_help_popover(
                        _HELP_ROLLOUT_TABLE.format(mod=mod),
                        key=f"t4_roll_{mod}_{roll_cohort}",
                    )
                st.dataframe(mini, hide_index=True, width="stretch")
