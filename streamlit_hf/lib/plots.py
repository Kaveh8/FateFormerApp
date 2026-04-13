"""Plotly helpers for the explorer UI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from streamlit_hf.lib.reactions import normalize_reaction_key

# Matches Streamlit theme primary + slate text; used across Plotly layouts.
PLOT_FONT = dict(family="Inter, system-ui, sans-serif", size=12)

PALETTE = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#db2777",
    "#0d9488",
    "#4f46e5",
)

MODALITY_COLOR = {"RNA": "#E64B35", "ATAC": "#4DBBD5", "Flux": "#00A087"}
# Global modality pie only: edit here to try other hues (bars/scatter use MODALITY_COLOR).
MODALITY_PIE_COLOR = dict(MODALITY_COLOR)
# Log₂FC heatmaps/sunburst: colours like ggplot2 scale_colour_gradient2 (mid grey at 0).
LOG_FC_COLOR_MIN = -0.5
LOG_FC_COLOR_MAX = 0.5
LOG_FC_DIVERGING_SCALE: list[list] = [
    [0.0, "#1C86EE"],
    [0.5, "#FAFAFA"],
    [1.0, "#FF0000"],
]
# Unicode minus (U+2212) and subscript ₁₀ / ₂ for axes/colorbars.
LABEL_NEG_LOG10_ADJ_P = "\u2212log\u2081\u2080 adj. p"
LABEL_LOG2FC = "Log\u2082FC"
# Cached attention dict uses lowercase modality keys.
FI_ATT_MOD_KEY = {"RNA": "rna", "ATAC": "atac", "Flux": "flux"}
# Model appends one batch-embedding token per modality; hide from attention rankings in the UI.
BATCH_EMBEDDING_FEATURE_NAMES = frozenset({"batch_rna", "batch_atac", "batch_flux"})


def _attention_pairs_skip_batch(pairs: list) -> list:
    return [(n, s) for n, s in pairs if str(n) not in BATCH_EMBEDDING_FEATURE_NAMES]


def rollout_top_features_table(feature_names, vec, top_n: int) -> pd.DataFrame:
    """Top `top_n` rollout weights per modality slice, excluding batch-embedding tokens."""
    names = [str(x) for x in feature_names]
    v = np.asarray(vec, dtype=float)
    rows = [
        (names[i], float(v[i]))
        for i in range(len(names))
        if names[i] not in BATCH_EMBEDDING_FEATURE_NAMES
    ]
    rows.sort(key=lambda x: -x[1])
    rows = rows[:top_n]
    if not rows:
        return pd.DataFrame(columns=["feature", "mean_attention"])
    feat, val = zip(*rows)
    return pd.DataFrame({"feature": list(feat), "mean_attention": list(val)})

# Themed continuous scale for dominant-fate % on UMAP (low → high emphasis).
UMAP_PCT_COLORSCALE: list[list] = [
    [0.0, "#eff6ff"],
    [0.25, "#bfdbfe"],
    [0.55, "#3b82f6"],
    [0.82, "#2563eb"],
    [1.0, "#1e3a8a"],
]

# Okabe–Ito–style distinct colours (colourblind-friendly) for categorical UMAP hues.
LATENT_DISCRETE_PALETTE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
)


def latent_scatter(
    df,
    color_col: str,
    title: str,
    width: int = 720,
    height: int = 520,
    marker_size: float = 5.0,
    marker_opacity: float = 0.78,
):
    d = df.copy()
    hover_spec = {
        "umap_x": ":.3f",
        "umap_y": ":.3f",
        "dataset_idx": True,
        "fold": True,
        "batch_no": True,
        "predicted_class": True,
        "label": True,
        "correct": True,
        "pct": ":.2f",
        "modality_label": True,
        "modality": True,
        "predicted_value": ":.3f",
        "clone_id": True,
        "clone_size": True,
        "cell_type": True,
    }
    if "modality_label" in d.columns:
        hover_spec.pop("modality", None)
    hover_data = {k: v for k, v in hover_spec.items() if k in d.columns}
    _disp = {
        "label": "CellTag-Multi label",
        "predicted_class": "Predicted fate",
        "pct": "Dominant fate (%)",
        "modality_label": "Available modalities",
        "dataset_idx": "Dataset index",
        "batch_no": "Batch",
        "fold": "CV fold",
    }
    labels_map = {c: _disp[c] for c in _disp if c in d.columns}

    continuous = color_col == "pct"
    if color_col == "fold":
        d["_color"] = d["fold"].astype(str)
        color_arg = "_color"
        labels_map["_color"] = "Fold"
        continuous = False
    elif color_col == "batch_no":
        d["_color"] = d["batch_no"].astype(str)
        color_arg = "_color"
        labels_map["_color"] = "Batch"
        continuous = False
    elif color_col == "correct":
        d["_color"] = d["correct"].map({True: "Correct", False: "Wrong"})
        color_arg = "_color"
        labels_map["_color"] = "Prediction"
        continuous = False
    else:
        color_arg = color_col

    common = dict(
        x="umap_x",
        y="umap_y",
        hover_data=hover_data,
        labels=labels_map,
        title=title,
        width=width,
        height=height,
    )
    if continuous:
        fig = px.scatter(
            d,
            color=color_arg,
            color_continuous_scale=UMAP_PCT_COLORSCALE,
            **common,
        )
    else:
        fig = px.scatter(
            d,
            color=color_arg,
            color_discrete_sequence=list(LATENT_DISCRETE_PALETTE),
            **common,
        )
    fig.update_traces(
        marker=dict(size=marker_size, opacity=marker_opacity, line=dict(width=0.25, color="rgba(255,255,255,0.4)"))
    )
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        title_font_size=16,
        margin=dict(l=28, r=20, t=56, b=28),
        legend_title_text="",
        xaxis_title="",
        yaxis_title="",
    )
    fig.update_xaxes(showticklabels=False, showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False)
    return fig


def rank_scatter_shift_vs_attention(df_mod, modality: str, width: int = 420, height: int = 440):
    """Attention rank on x, shift rank on y, least-squares trend line, discrete point colours."""
    need = ("shift_order_mod", "attention_order_mod")
    if not all(c in df_mod.columns for c in need):
        return go.Figure()
    sub = df_mod.dropna(subset=list(need)).copy()
    if sub.empty:
        return go.Figure()
    x = sub["attention_order_mod"].astype(float).to_numpy()
    y = sub["shift_order_mod"].astype(float).to_numpy()
    fig = px.scatter(
        sub,
        x="attention_order_mod",
        y="shift_order_mod",
        color="top_10_pct",
        hover_name="feature",
        hover_data={
            "mean_rank": True,
            "importance_shift": ":.4f",
            "importance_att": ":.4f",
        },
        labels={
            "attention_order_mod": "Attention rank",
            "shift_order_mod": "Shift rank",
        },
        width=width,
        height=height,
        color_discrete_map={
            "both": PALETTE[0],
            "shift": PALETTE[1],
            "att": PALETTE[2],
            "None": "#94a3b8",
        },
    )
    fig.update_traces(marker=dict(size=7, opacity=0.62, line=dict(width=0.5, color="rgba(15,23,42,0.28)")))
    if len(x) >= 2 and float(np.ptp(x)) > 0:
        coef = np.polyfit(x, y, 1)
        poly = np.poly1d(coef)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=poly(xs),
                mode="lines",
                name=f"y = {coef[0]:.2f}x + {coef[1]:.2f}",
                line=dict(color="#2563eb", width=2, dash="dash"),
                showlegend=True,
            )
        )
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        title=dict(
            text=f"{modality}: shift vs attention (ranks)",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=14, family=PLOT_FONT["family"]),
        ),
        margin=dict(l=48, r=20, t=52, b=72),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


def _truncate_label(s: str, max_len: int = 36) -> str:
    s = str(s)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def joint_shift_attention_top_features(df_mod, modality: str, top_n: int):
    """
    Top features by mean_rank (lowest = strongest joint shift+attention ranking).
    Shift and attention importances are min–max scaled within this top-N slice for side-by-side comparison.
    """
    need = ("mean_rank", "importance_shift", "importance_att", "feature")
    if not all(c in df_mod.columns for c in need):
        return go.Figure()
    sub = df_mod.nsmallest(top_n, "mean_rank").copy()
    if sub.empty:
        return go.Figure()

    def _mm(s: pd.Series) -> pd.Series:
        lo, hi = float(s.min()), float(s.max())
        if hi <= lo:
            return pd.Series(0.5, index=s.index)
        return (s.astype(float) - lo) / (hi - lo)

    sub["_zs"] = _mm(sub["importance_shift"])
    sub["_za"] = _mm(sub["importance_att"])
    # Best (lowest mean_rank) at top of chart; matches shift/attention rows below.
    sub = sub.sort_values("mean_rank", ascending=True)
    feats_full = sub["feature"].astype(str)
    y_disp = feats_full.map(lambda s: _truncate_label(s, 40))
    base = MODALITY_COLOR.get(modality, PALETTE[0])
    att_c = "#475569" if base != "#475569" else "#64748b"

    margin_l = int(min(380, 64 + 5.8 * max((len(t) for t in y_disp), default=10)))
    h = min(720, 52 + 22 * len(sub))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Shift (scaled)",
            y=y_disp,
            x=sub["_zs"],
            orientation="h",
            marker_color=base,
            customdata=feats_full,
            hovertemplate="<b>%{customdata}</b><br>Shift (scaled): %{x:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Attention (scaled)",
            y=y_disp,
            x=sub["_za"],
            orientation="h",
            marker_color=att_c,
            customdata=feats_full,
            hovertemplate="<b>%{customdata}</b><br>Attention (scaled): %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        title=dict(
            text=f"{modality} · top {top_n}",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=14, family=PLOT_FONT["family"]),
        ),
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
        width=680,
        height=h,
        margin=dict(l=margin_l, r=12, t=44, b=72),
        xaxis_title="Scaled 0-1 within selection",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5),
    )
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    return fig


def modality_shift_attention_rank_stats(df_mod) -> dict[str, Any]:
    """Pearson / Spearman between per-modality shift and attention ordinal ranks."""
    from scipy.stats import pearsonr, spearmanr

    need = ("shift_order_mod", "attention_order_mod")
    if not all(c in df_mod.columns for c in need):
        return {"n": 0}
    sub = df_mod.dropna(subset=list(need))
    n = len(sub)
    if n < 3:
        return {"n": n}
    xs = sub["attention_order_mod"].astype(float)
    ys = sub["shift_order_mod"].astype(float)
    pr, pp = pearsonr(xs, ys)
    sr, sp = spearmanr(xs, ys)
    return {
        "n": n,
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


def rank_bar(
    df_top,
    xcol: str,
    ycol: str,
    title: str,
    color: str = PALETTE[0],
    xaxis_title: str | None = None,
):
    d = df_top.sort_values(xcol, ascending=True)
    y_raw = d[ycol].astype(str)
    y_show = y_raw.map(lambda s: _truncate_label(s, 42))
    margin_l = int(min(420, 80 + 5.8 * max((len(s) for s in y_show), default=12)))
    fig = go.Figure(
        go.Bar(
            y=y_show,
            x=d[xcol],
            orientation="h",
            marker_color=color,
            customdata=y_raw,
            hovertemplate="<b>%{customdata}</b><br>%{x:.4g}<extra></extra>",
        )
    )
    xt = xaxis_title if xaxis_title is not None else xcol.replace("_", " ")
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        title=title,
        width=680,
        height=min(620, 38 + 20 * len(d)),
        margin=dict(l=margin_l, r=24, t=48, b=40),
        xaxis_title=xt,
        yaxis_title="",
    )
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


def attention_top_comparison(fi_lists: dict, modality: str, top_n: int = 18):
    """fi_lists: cohort -> {rna|atac|flux: [(name, score), ...]}."""
    mk = FI_ATT_MOD_KEY.get(modality, str(modality).lower())
    traces = []
    for key, name, color in (
        ("all", "All validation samples", PALETTE[0]),
        ("dead_end", "Predicted dead-end", PALETTE[1]),
        ("reprogramming", "Predicted reprogramming", PALETTE[2]),
    ):
        cohort = fi_lists.get(key) or {}
        items = _attention_pairs_skip_batch(list(cohort.get(mk, [])))[:top_n]
        if not items:
            continue
        feats, scores = zip(*items)
        traces.append(
            go.Bar(
                name=name,
                x=list(scores),
                y=[f[:52] + ("…" if len(f) > 52 else "") for f in feats],
                orientation="h",
                marker_color=color,
            )
        )
    fig = go.Figure(traces)
    bar_h = max(320, 36 + min(top_n, 20) * 22 * max(1, len(traces)))
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        font=PLOT_FONT,
        title=f"Top attention (rollout): {modality}",
        width=520,
        height=bar_h,
        margin=dict(l=220, r=24, t=56, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if not traces:
        fig.update_layout(
            annotations=[
                dict(
                    text="No attention list for this modality (re-run precompute).",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
            ]
        )
    else:
        fig.update_yaxes(autorange="reversed")
    return fig


def attention_cohort_view(
    fi_lists: dict,
    modality: str,
    top_n: int,
    mode: str,
):
    """
    mode: 'compare': grouped bars for all three cohorts;
          'all' | 'dead_end' | 'reprogramming': single cohort only.
    """
    if mode == "compare":
        return attention_top_comparison(fi_lists, modality, top_n)
    mk = FI_ATT_MOD_KEY.get(modality, str(modality).lower())
    cohort = fi_lists.get(mode) or {}
    items = _attention_pairs_skip_batch(list(cohort.get(mk, [])))[:top_n]
    label = {
        "all": "All validation samples",
        "dead_end": "Predicted dead-end",
        "reprogramming": "Predicted reprogramming",
    }.get(mode, mode)
    if not items:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            font=PLOT_FONT,
            title=f"{modality} · {label}",
            annotations=[
                dict(
                    text="No items for this cohort.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
            ],
        )
        return fig
    feats, scores = zip(*items)
    fig = go.Figure(
        go.Bar(
            x=list(scores),
            y=[f[:52] + ("…" if len(f) > 52 else "") for f in feats],
            orientation="h",
            marker_color=PALETTE[0],
        )
    )
    h = max(280, 40 + min(top_n, 25) * 20)
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        title=f"{modality} · {label}",
        width=520,
        height=h,
        margin=dict(l=220, r=24, t=56, b=40),
        xaxis_title="Attention weight",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def global_rank_triple_panel(df_features, top_n: int = 20, top_n_pie: int = 100):
    """
    Global top-N by latent-shift and by attention (min–max scaled), plus pie of modality mix
    among the top `top_n_pie` features by mean rank.
    """
    d = df_features.copy()
    for col in ("importance_shift", "importance_att"):
        min_v, max_v = d[col].min(), d[col].max()
        if max_v > min_v:
            d[col + "_norm"] = (d[col] - min_v) / (max_v - min_v)
        else:
            d[col + "_norm"] = 0.0

    shift_top = d.nlargest(top_n, "importance_shift")
    att_top = d.nlargest(top_n, "importance_att")
    pie_pool = d.nsmallest(top_n_pie, "mean_rank")

    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.36, 0.36, 0.28],
        specs=[[{}, {}, {"type": "domain"}]],
        subplot_titles=(
            f"Top {top_n} by latent shift (ranked)",
            f"Top {top_n} by attention (ranked)",
            f"Top {top_n_pie} by mean rank (modality mix)",
        ),
        horizontal_spacing=0.06,
    )

    fig.add_trace(
        go.Bar(
            x=shift_top["importance_shift_norm"],
            y=shift_top["feature"],
            orientation="h",
            marker_color=[MODALITY_COLOR.get(m, "#64748b") for m in shift_top["modality"]],
            marker_line=dict(color="rgba(15,23,42,0.12)", width=1),
            showlegend=False,
            hovertemplate="%{y}<br>scaled shift: %{x:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=att_top["importance_att_norm"],
            y=att_top["feature"],
            orientation="h",
            marker_color=[MODALITY_COLOR.get(m, "#64748b") for m in att_top["modality"]],
            marker_line=dict(color="rgba(15,23,42,0.12)", width=1),
            showlegend=False,
            hovertemplate="%{y}<br>scaled attention: %{x:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    pie_labels = ["RNA", "ATAC", "Flux"]
    counts = pie_pool["modality"].value_counts()
    pie_vals = [int(counts.get(lab, 0)) for lab in pie_labels]
    if sum(pie_vals) == 0:
        pie_vals = [1, 1, 1]

    fig.add_trace(
        go.Pie(
            labels=pie_labels,
            values=pie_vals,
            marker=dict(
                colors=[MODALITY_PIE_COLOR.get(l, "#64748b") for l in pie_labels],
                line=dict(color="#1e293b", width=1.2),
            ),
            textinfo="label+percent",
            textfont_size=12,
            hole=0.0,
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.update_xaxes(title_text="Min-max scaled shift", row=1, col=1)
    fig.update_xaxes(title_text="Min-max scaled attention", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)

    h = max(480, 40 + top_n * 18)
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        height=h,
        width=min(1280, 400 + top_n * 14),
        margin=dict(l=40, r=40, t=80, b=40),
        title_text="Global feature ranking (all modalities)",
        title_x=0.5,
    )
    return fig


def _flux_prepare_top_ranked(flux_df: pd.DataFrame, top_n: int, metric: str = "mean_rank") -> pd.DataFrame:
    sub = flux_df[~flux_df["feature"].astype(str).str.contains("batch", case=False, na=False)].copy()
    if metric not in sub.columns:
        metric = "mean_rank"
    sub = sub.sort_values(metric, ascending=True).head(int(top_n)).copy()
    if "pathway" in sub.columns:
        pc = sub["pathway"].value_counts()
        sub["_pw_n"] = sub["pathway"].map(pc)
        sub.sort_values(["_pw_n", "pathway"], ascending=[False, True], inplace=True)
    return sub


def flux_pathway_sunburst(flux_df: pd.DataFrame, max_features: int = 55) -> go.Figure:
    sub = flux_df.dropna(subset=["pathway"]).copy()
    if sub.empty:
        return go.Figure()
    sub = sub.nsmallest(int(max_features), "mean_rank")
    sub["pathway"] = sub["pathway"].astype(str)
    sub["_uid"] = np.arange(len(sub))
    sub["rxn"] = sub.apply(
        lambda r: f"{_truncate_label(str(r['feature']), 36)} ·{int(r['_uid'])}",
        axis=1,
    )
    mr = sub["mean_rank"].astype(float)
    sub["w"] = (mr.max() - mr + 1.0).clip(lower=0.5)
    color_col = "log_fc" if "log_fc" in sub.columns and sub["log_fc"].notna().any() else "mean_rank"
    sb_kw: dict[str, Any] = {
        "path": ["pathway", "rxn"],
        "values": "w",
        "color": color_col,
        "hover_data": {"mean_rank": ":.2f", "pval_adj": ":.2e", "feature": True, "w": False, "_uid": False},
    }
    if color_col == "log_fc":
        sb_kw["color_continuous_scale"] = LOG_FC_DIVERGING_SCALE
        sb_kw["range_color"] = [LOG_FC_COLOR_MIN, LOG_FC_COLOR_MAX]
    else:
        sb_kw["color_continuous_scale"] = "Viridis_r"
    fig = px.sunburst(sub, **sb_kw)
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        margin=dict(l=8, r=8, t=100, b=16),
        height=min(820, 520 + int(max_features) * 5),
        title=dict(
            text="Top flux reactions by model rank, nested under pathway",
            x=0,
            xanchor="left",
            y=0.99,
            yanchor="top",
            font=dict(size=13, family=PLOT_FONT["family"]),
            pad=dict(b=16, l=4),
        ),
    )
    if color_col == "log_fc":
        fig.update_layout(
            coloraxis=dict(
                cmin=LOG_FC_COLOR_MIN,
                cmax=LOG_FC_COLOR_MAX,
                colorbar=dict(
                    title=dict(text=LABEL_LOG2FC, side="right"),
                    tickformat=".2f",
                    len=0.38,
                    thickness=12,
                    y=0.52,
                    yanchor="middle",
                ),
            )
        )
    return fig


def flux_volcano(flux_df: pd.DataFrame) -> go.Figure:
    if "log_fc" not in flux_df.columns:
        return go.Figure()
    d = flux_df.dropna(subset=["log_fc"]).copy()
    if d.empty:
        return go.Figure()
    # Drop degenerate rows: ~zero fold-change with exactly-zero adjusted p (numeric artifact / noise).
    lf = d["log_fc"].astype(float)
    if "pval_adj" in d.columns:
        pa = d["pval_adj"].astype(float)
        bad = np.isfinite(lf) & np.isfinite(pa) & (np.abs(lf) < 1e-10) & (pa <= 0.0)
        d = d[~bad]
    if d.empty:
        return go.Figure()
    if "pval_adj_log" in d.columns:
        y = d["pval_adj_log"].astype(float)
    else:
        p = d["pval_adj"].astype(float).clip(lower=1e-300)
        y = -np.log10(p.to_numpy())
    d = d.assign(_neglogp=y)
    fig = px.scatter(
        d,
        x="log_fc",
        y="_neglogp",
        color="mean_rank",
        color_continuous_scale="Viridis_r",
        hover_name="feature",
        hover_data=["pathway", "pval_adj", "group"],
        labels={
            "log_fc": LABEL_LOG2FC,
            "_neglogp": LABEL_NEG_LOG10_ADJ_P,
            "mean_rank": "Mean rank",
        },
    )
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        title="Differential flux vs statistical significance",
        height=520,
        margin=dict(l=52, r=24, t=52, b=48),
        coloraxis_colorbar=dict(
            title=dict(text="Mean rank", side="right"),
            thickness=12,
            len=0.55,
        ),
    )
    return fig


def motif_tf_mean_rank_bars(atac_df: pd.DataFrame, top_n: int = 22) -> go.Figure:
    """Aggregate motif features by TF name (prefix before ``_<motif_id>``); show lowest mean joint rank."""
    if atac_df.empty or "feature" not in atac_df.columns:
        return go.Figure()

    def _tf_prefix(feat: str) -> str:
        s = str(feat)
        if "_" in s:
            head, tail = s.rsplit("_", 1)
            if tail.isdigit():
                return head
        return s

    d = atac_df.copy()
    d["_tf"] = d["feature"].map(_tf_prefix)
    agg = d.groupby("_tf", as_index=False)["mean_rank"].mean()
    agg = agg.nsmallest(int(top_n), "mean_rank").sort_values("mean_rank", ascending=True)
    if agg.empty:
        return go.Figure()
    y_show = agg["_tf"].astype(str).map(lambda s: _truncate_label(s, 36))
    fig = go.Figure(
        go.Bar(
            y=y_show,
            x=agg["mean_rank"],
            orientation="h",
            marker_color=MODALITY_COLOR.get("ATAC", PALETTE[0]),
            customdata=agg["_tf"],
            hovertemplate="<b>%{customdata}</b><br>Mean mean_rank (across motifs): %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        title=f"TFs by average motif rank (top {top_n} by lowest mean rank)",
        height=min(640, 48 + 22 * len(agg)),
        margin=dict(l=160, r=24, t=52, b=40),
        xaxis_title="Mean of mean_rank over motif instances (lower = stronger)",
        yaxis_title="",
    )
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    return fig


def motif_chromvar_volcano(atac_df: pd.DataFrame) -> go.Figure:
    """Motif differential view: mean activity difference (reprogramming − dead-end) vs significance."""
    need = ("mean_diff", "pval_adj")
    if not all(c in atac_df.columns for c in need):
        return go.Figure()
    d = atac_df.dropna(subset=["mean_diff", "pval_adj"]).copy()
    if d.empty:
        return go.Figure()
    md = d["mean_diff"].astype(float)
    pa = d["pval_adj"].astype(float)
    bad = np.isfinite(md) & np.isfinite(pa) & (np.abs(md) < 1e-12) & (pa <= 0.0)
    d = d[~bad]
    if d.empty:
        return go.Figure()
    if "pval_adj_log" in d.columns:
        y = d["pval_adj_log"].astype(float)
    else:
        p = d["pval_adj"].astype(float).clip(lower=1e-300)
        y = -np.log10(p.to_numpy())
    d = d.assign(_y=y)
    hover_cols = [c for c in ("group", "pval_adj", "mean_rank", "mean_de", "mean_re") if c in d.columns]
    fig = px.scatter(
        d,
        x="mean_diff",
        y="_y",
        color="mean_rank",
        color_continuous_scale="Viridis_r",
        hover_name="feature",
        hover_data=hover_cols if hover_cols else None,
        labels={
            "mean_diff": "Mean difference (reprogramming − dead-end)",
            "_y": LABEL_NEG_LOG10_ADJ_P,
            "mean_rank": "Mean rank",
        },
    )
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        title="TF motif differential activity (mean difference vs significance)",
        height=520,
        margin=dict(l=52, r=24, t=52, b=48),
        coloraxis_colorbar=dict(title=dict(text="Mean rank", side="right"), thickness=12, len=0.55),
    )
    return fig


def notebook_style_activity_scatter(
    df: pd.DataFrame,
    title: str,
    x_title: str,
    y_title: str,
) -> go.Figure:
    """mean_de vs mean_re, colour = pval_adj_log (Reds), marker size ∝ inverse mean_rank."""
    need = ("mean_de", "mean_re", "mean_rank", "pval_adj_log", "feature", "group")
    if not all(c in df.columns for c in need):
        return go.Figure()
    d = df.dropna(subset=["mean_de", "mean_re", "mean_rank", "pval_adj_log"]).copy()
    if d.empty:
        return go.Figure()
    mx = float(d["mean_rank"].max())
    d = d.assign(_inv=(mx - d["mean_rank"].astype(float)).clip(lower=0))
    inv = d["_inv"].astype(float)
    lo, hi = float(inv.min()), float(inv.max())
    if hi <= lo:
        d["_sz"] = 6.0
    else:
        d["_sz"] = 3.5 + (inv - lo) / (hi - lo) * 9.0

    fig = px.scatter(
        d,
        x="mean_de",
        y="mean_re",
        color="pval_adj_log",
        color_continuous_scale="Reds",
        size="_sz",
        size_max=14,
        hover_name="feature",
        hover_data={
            "mean_rank": ":.2f",
            "group": True,
            "pval_adj_log": ":.2f",
            "_inv": False,
            "_sz": False,
        },
        labels={
            "mean_de": x_title,
            "mean_re": y_title,
            "pval_adj_log": "Adj. p-value (log)",
        },
    )
    fig.update_traces(
        marker=dict(line=dict(width=0.45, color="rgba(255,255,255,0.75)"), opacity=0.9),
        selector=dict(mode="markers"),
    )
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        title=title,
        height=520,
        margin=dict(l=52, r=24, t=52, b=48),
        coloraxis_colorbar=dict(title=dict(text="Adj. p (log)", side="right"), thickness=12, len=0.55),
    )
    return fig


def pathway_bubble_suggested_height(n_paths: int) -> int:
    """Total figure height for pathway bubble panels (use the max of both cohorts so legends line up)."""
    n = max(int(n_paths), 1)
    return max(520, min(1100, 22 * n + 200))


def pathway_enrichment_bubble_panel(
    df: pd.DataFrame,
    title: str,
    *,
    show_colorbar: bool = True,
    layout_height: int | None = None,
) -> go.Figure:
    """Single cohort: Reactome (circle) vs KEGG (square), colour = −log₁₀ Benjamini (scale per panel)."""
    fig = go.Figure()
    if df.empty:
        fig.update_layout(
            template="plotly_white",
            font=PLOT_FONT,
            title=dict(text=title, x=0.5, xanchor="center"),
            annotations=[
                dict(
                    text="No significant pathways (Benjamini–Hochberg q < 0.05)",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=13, color="#64748b"),
                )
            ],
            height=320,
            margin=dict(l=40, r=40, t=56, b=40),
        )
        return fig

    # More genes in the overlap first, then stronger gene ratio (matches enrichment table emphasis).
    d = df.sort_values(by=["Count", "Gene Ratio"], ascending=[False, False]).reset_index(drop=True)
    d = d.assign(
        _neglog=-np.log10(d["Benjamini"].astype(float).clip(lower=1e-300)),
        _y=np.arange(len(d), dtype=float),
    )
    nl = d["_neglog"].astype(float)
    cmin = float(nl.min())
    cmax = float(nl.max())
    if cmax <= cmin:
        cmax = cmin + 1e-6

    # Single trace: per-panel cmin/cmax so Viridis uses the cohort’s range (shared global max clusters at one hue).
    sym_map = {"Reactome": "circle", "KEGG": "square"}
    symbols = [sym_map.get(str(x), "circle") for x in d["Library"].tolist()]
    sz = np.sqrt(d["Count"].astype(float).clip(lower=1)) * 4.8
    customdata = np.stack(
        [d["Count"].to_numpy(), d["_neglog"].to_numpy(), d["Library"].astype(str).to_numpy()],
        axis=1,
    )
    fig.add_trace(
        go.Scatter(
            x=d["Gene Ratio"],
            y=d["_y"],
            mode="markers",
            name="Pathways",
            showlegend=False,
            marker=dict(
                size=sz,
                sizemode="diameter",
                sizemin=4,
                symbol=symbols,
                color=d["_neglog"],
                cmin=cmin,
                cmax=cmax,
                colorscale="Viridis",
                showscale=bool(show_colorbar),
                colorbar=dict(
                    title=dict(
                        text="\u2212log\u2081\u2080 q",
                        side="right",
                    ),
                    len=0.72,
                    thickness=12,
                    y=0.45,
                    yanchor="middle",
                    outlinewidth=0,
                )
                if show_colorbar
                else None,
                line=dict(width=0.75, color="rgba(0,0,0,0.5)"),
                opacity=0.92,
            ),
            text=d["Term"],
            customdata=customdata,
            hovertemplate=(
                "<b>%{text}</b><br>%{customdata[2]}<br>Gene ratio: %{x:.3f}<br>Count: %{customdata[0]}"
                "<br>\u2212log\u2081\u2080 Benjamini: %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )
    for lib, sym in (("Reactome", "circle"), ("KEGG", "square")):
        if lib not in set(d["Library"].astype(str)):
            continue
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=lib,
                marker=dict(
                    symbol=sym,
                    size=11,
                    color="#475569",
                    line=dict(width=1, color="rgba(0,0,0,0.45)"),
                ),
                showlegend=True,
            )
        )

    ticktext = [_truncate_label(str(t), 52) for t in d["Term"]]
    h = int(layout_height) if layout_height is not None else pathway_bubble_suggested_height(len(d))
    fig.update_yaxes(
        tickmode="array",
        tickvals=d["_y"].tolist(),
        ticktext=ticktext,
        autorange="reversed",
        title="",
    )
    fig.update_xaxes(title_text="Gene ratio (count ÷ list total)")
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            yanchor="top",
            y=0.985,
            pad=dict(b=0),
        ),
        height=h,
        margin=dict(l=215, r=132, t=48, b=108),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.11,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
        ),
        showlegend=True,
    )
    return fig


def pathway_gene_membership_heatmap(
    z: np.ndarray, row_labels: list[str], col_labels: list[str]
) -> go.Figure:
    """Pathway × gene grid; empty cells transparent; light gaps; legend for category colours."""
    if z.size == 0:
        return go.Figure()
    # Discrete codes 0–4 must not use z/4 (3→0.75 landed in the KEGG band). Map to fixed slots.
    _z_plot = {0: 0.04, 1: 0.24, 2: 0.44, 3: 0.64, 4: 0.84}
    zn = np.vectorize(lambda v: _z_plot.get(int(v), 0.04))(z).astype(float)
    transparent = "rgba(0,0,0,0)"
    colorscale = [
        [0.0, transparent],
        [0.14, transparent],
        [0.15, "#e69138"],
        [0.33, "#e69138"],
        [0.34, "#7eb6d9"],
        [0.53, "#7eb6d9"],
        [0.54, "#9ccc65"],
        [0.73, "#9ccc65"],
        [0.74, "#283593"],
        [1.0, "#283593"],
    ]

    def _cell_hint(v: float) -> str:
        k = int(round(float(v)))
        return {
            0: "",
            1: "Gene enriched in dead-end contrast",
            2: "Gene enriched in reprogramming contrast",
            3: "Reactome pathway set",
            4: "KEGG pathway set",
        }.get(k, "")

    z_int = z.astype(int)
    text_grid = [[_cell_hint(z_int[i, j]) for j in range(z.shape[1])] for i in range(z.shape[0])]

    heat = go.Heatmap(
        z=zn,
        x=col_labels,
        y=row_labels,
        text=text_grid,
        colorscale=colorscale,
        zmin=0,
        zmax=1,
        showscale=False,
        xgap=1,
        ygap=1,
        hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>",
    )

    fig = go.Figure(data=[heat])

    n_rows, n_cols = z.shape
    cell_w = 10
    cell_h = 20
    w = int(min(1000, max(460, n_cols * cell_w + 272)))
    h = int(min(960, max(460, n_rows * cell_h + 128)))
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        title=dict(text="Pathway–gene membership", x=0.5, xanchor="center"),
        height=h,
        width=w,
        margin=dict(l=4, r=168, t=52, b=108),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f4f6f9",
        xaxis=dict(side="bottom", tickangle=-50, showgrid=False, zeroline=False),
        yaxis=dict(
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=False,
            autorange="reversed",
        ),
    )

    legend_markers = [
        ("Empty cell", "#f1f5f9", "square"),
        ("Dead-end–linked gene", "#e69138", "square"),
        ("Reprogramming–linked gene", "#7eb6d9", "square"),
        ("Reactome (column tag)", "#9ccc65", "square"),
        ("KEGG (column tag)", "#283593", "square"),
    ]
    for name, color, sym in legend_markers:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=name,
                marker=dict(size=11, color=color, symbol=sym, line=dict(width=1, color="rgba(0,0,0,0.25)")),
                showlegend=True,
            )
        )

    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
            font=dict(size=11),
        )
    )
    return fig


def flux_dead_end_vs_reprogram_scatter(flux_df: pd.DataFrame, max_pathway_colors: int = 12) -> go.Figure:
    need = ("mean_de", "mean_re")
    if not all(c in flux_df.columns for c in need):
        return go.Figure()
    d = flux_df.dropna(subset=list(need)).copy()
    if d.empty:
        return go.Figure()
    imp = (
        d["importance_shift"].astype(float).clip(lower=0) * d["importance_att"].astype(float).clip(lower=0)
    ) ** 0.5
    q = float(imp.quantile(0.95)) if len(imp) else 1.0
    d = d.assign(_s=(imp / (q or 1.0)).clip(upper=1) * 20 + 5)
    pw = d["pathway"].fillna("Unknown").astype(str) if "pathway" in d.columns else pd.Series(
        ["Unknown"] * len(d), index=d.index
    )
    top_pw = pw.value_counts().head(int(max_pathway_colors)).index
    d = d.assign(_pw_col=pw.where(pw.isin(top_pw), "Other"))
    uniq = sorted(d["_pw_col"].astype(str).unique(), key=lambda x: (x == "Other", x))
    pal = list(LATENT_DISCRETE_PALETTE)
    pw_cmap: dict[str, str] = {}
    j = 0
    for name in uniq:
        if name == "Other":
            pw_cmap[name] = "#94a3b8"
        else:
            pw_cmap[name] = pal[j % len(pal)]
            j += 1
    fig = px.scatter(
        d,
        x="mean_de",
        y="mean_re",
        color="_pw_col",
        color_discrete_map=pw_cmap,
        size="_s",
        hover_name="feature",
        hover_data=["mean_rank", "log_fc", "pathway"],
        labels={
            "mean_de": "Mean flux · dead-end",
            "mean_re": "Mean flux · reprogramming",
            "_pw_col": "Pathway",
        },
    )
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        height=540,
        margin=dict(l=52, r=20, t=52, b=40),
        title="Average measured flux by fate label (each point is one reaction)",
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5),
    )
    fig.update_traces(marker=dict(opacity=0.75, line=dict(width=0.35, color="rgba(0,0,0,0.3)")))
    return fig


def flux_pathway_mean_rank_violin(flux_df: pd.DataFrame, top_pathways: int = 12) -> go.Figure:
    sub = flux_df.dropna(subset=["pathway"]).copy()
    if sub.empty:
        return go.Figure()
    top_p = sub["pathway"].astype(str).value_counts().head(int(top_pathways)).index
    sub = sub[sub["pathway"].astype(str).isin(top_p)]
    top_list = list(top_p)
    v_cmap = {p: LATENT_DISCRETE_PALETTE[i % len(LATENT_DISCRETE_PALETTE)] for i, p in enumerate(top_list)}
    fig = px.violin(
        sub,
        x="pathway",
        y="mean_rank",
        box=True,
        points=False,
        color="pathway",
        color_discrete_map=v_cmap,
        labels={"mean_rank": "Mean rank (lower = stronger model focus)", "pathway": "Pathway"},
    )
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        showlegend=False,
        height=420,
        xaxis_tickangle=-32,
        margin=dict(l=48, r=24, t=48, b=140),
        title="How joint model rank spreads within high-coverage pathways",
    )
    return fig


def flux_reaction_annotation_panel(flux_df: pd.DataFrame, top_n: int = 26, metric: str = "mean_rank") -> go.Figure:
    """Three heatmap columns: pathway (categorical), DE Log₂FC, −log₁₀ adjusted p."""
    top = _flux_prepare_top_ranked(flux_df, top_n, metric)
    if top.empty:
        return go.Figure()
    n = len(top)
    pathways = top["pathway"].fillna("Unknown").astype(str).tolist() if "pathway" in top.columns else ["Unknown"] * n
    uniq = list(dict.fromkeys(pathways))
    code_map = {u: i for i, u in enumerate(uniq)}
    codes = np.array([code_map[p] for p in pathways], dtype=float)
    k = max(len(uniq), 1)
    qual = list(px.colors.qualitative.Safe) + list(px.colors.qualitative.Dark24) + list(px.colors.qualitative.Light24)
    if k <= 1:
        disc_scale = [[0, qual[0]], [1, qual[0]]]
    else:
        disc_scale = [[j / (k - 1), qual[j % len(qual)]] for j in range(k)]
    log_fc = top["log_fc"].fillna(0).astype(float).to_numpy() if "log_fc" in top.columns else np.zeros(n)
    if "pval_adj_log" in top.columns:
        pv = top["pval_adj_log"].fillna(0).astype(float).to_numpy()
    else:
        pv = -np.log10(top["pval_adj"].astype(float).clip(lower=1e-300).to_numpy())
    full_features = top["feature"].astype(str).tolist()
    y_labels = [_truncate_label(str(f), 44) for f in full_features]
    z_path = codes.reshape(-1, 1)
    # hovertext (not customdata): subplot heatmaps often render %{customdata[0]} as "-" in the browser.
    hover_path = [[f"<b>{fn}</b><br>pathway: {pw}"] for fn, pw in zip(full_features, pathways)]
    hover_lfc = [
        [f"<b>{fn}</b><br>{LABEL_LOG2FC}: {float(log_fc[i]):.4f}"]
        for i, fn in enumerate(full_features)
    ]
    hover_pv = [
        [f"<b>{fn}</b><br>{LABEL_NEG_LOG10_ADJ_P}: {float(pv[i]):.2f}"]
        for i, fn in enumerate(full_features)
    ]
    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        column_widths=[0.24, 0.24, 0.24],
    )
    fig.add_trace(
        go.Heatmap(
            z=z_path,
            x=[""],
            y=y_labels,
            colorscale=disc_scale,
            zmin=0,
            zmax=max(k - 1, 0),
            showscale=False,
            hovertext=hover_path,
            hovertemplate="%{hovertext}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=log_fc.reshape(-1, 1),
            x=[""],
            y=y_labels,
            colorscale=LOG_FC_DIVERGING_SCALE,
            zmin=LOG_FC_COLOR_MIN,
            zmax=LOG_FC_COLOR_MAX,
            showscale=True,
            colorbar=dict(
                title=dict(text=LABEL_LOG2FC, side="right"),
                tickformat=".2f",
                len=0.22,
                y=0.71,
                yanchor="middle",
                x=1.0,
                xanchor="left",
                xref="paper",
                yref="paper",
                thickness=12,
            ),
            hovertext=hover_lfc,
            hovertemplate="%{hovertext}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            z=pv.reshape(-1, 1),
            x=[""],
            y=y_labels,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title=dict(text=LABEL_NEG_LOG10_ADJ_P, side="right"),
                len=0.22,
                y=0.29,
                yanchor="middle",
                x=1.0,
                xanchor="left",
                xref="paper",
                yref="paper",
                thickness=12,
            ),
            hovertext=hover_pv,
            hovertemplate="%{hovertext}<extra></extra>",
        ),
        row=1,
        col=3,
    )
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        height=min(820, 120 + n * 22),
        width=900,
        margin=dict(l=8, r=108, t=56, b=72),
        title=dict(
            text=f"Pathway, {LABEL_LOG2FC}, and significance",
            x=0,
            xanchor="left",
            y=0.995,
            yanchor="top",
            font=dict(size=13, family=PLOT_FONT["family"]),
            pad=dict(b=8, l=4),
        ),
    )
    fig.update_xaxes(side="bottom", title_standoff=8)
    fig.update_xaxes(title_text="Pathway", row=1, col=1)
    fig.update_xaxes(title_text=LABEL_LOG2FC, row=1, col=2)
    fig.update_xaxes(title_text=LABEL_NEG_LOG10_ADJ_P, row=1, col=3)
    fig.update_yaxes(autorange="reversed")
    return fig


def flux_model_metric_profile(flux_df: pd.DataFrame, top_n: int = 22, metric: str = "mean_rank") -> go.Figure:
    """Matrix view: scaled shift, attention, model priority, and fate flux contrast."""
    top = _flux_prepare_top_ranked(flux_df, top_n, metric)
    if top.empty:
        return go.Figure()

    def mm(s: pd.Series) -> np.ndarray:
        v = s.astype(float).to_numpy()
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if hi <= lo or not np.isfinite(lo):
            return np.zeros_like(v, dtype=float)
        return (v - lo) / (hi - lo)

    cols: list[np.ndarray] = []
    labels: list[str] = []
    for c, lab in (("importance_shift", "Latent shift impact"), ("importance_att", "Attention (rollout)")):
        if c in top.columns:
            cols.append(mm(top[c]))
            labels.append(lab)
    cols.append(1.0 - mm(top["mean_rank"]))
    labels.append("Joint priority (1 - scaled mean rank)")
    if "mean_de" in top.columns and "mean_re" in top.columns:
        de = top["mean_de"].astype(float).replace(0, np.nan)
        ratio = (top["mean_re"].astype(float) / (de + 1e-12)).fillna(0)
        cols.append(mm(ratio))
        labels.append("RE / DE mean flux (scaled)")
    z = np.column_stack(cols)
    full_rxn = top["feature"].astype(str).tolist()
    x_labels = [_truncate_label(str(f), 34) for f in full_rxn]
    fig = px.imshow(
        z.T,
        x=x_labels,
        y=labels,
        aspect="auto",
        color_continuous_scale="Tealrose",
        labels=dict(x="Reaction", y="Metric", color="Scaled 0-1 per metric"),
    )
    n_met, n_rxn = z.T.shape
    hover_cd = np.broadcast_to(np.array(full_rxn, dtype=object), (n_met, n_rxn))
    fig.update_traces(
        customdata=hover_cd,
        hovertemplate="<b>%{customdata}</b><br>%{y}<br>scaled: %{z:.3f}<extra></extra>",
    )
    fig.update_xaxes(tickangle=-50, side="bottom", title_standoff=12)
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        height=min(380, 140 + len(labels) * 36),
        margin=dict(l=200, r=28, t=64, b=200),
        title=dict(
            text="Reaction profile",
            x=0,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(size=13, family=PLOT_FONT["family"]),
            pad=dict(b=10, l=4),
        ),
    )
    return fig


