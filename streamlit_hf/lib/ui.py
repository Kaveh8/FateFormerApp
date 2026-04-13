"""Light shared styles (no heavy themes; keeps default Streamlit + plotly_white)."""

from __future__ import annotations

import streamlit as st


def inject_app_styles() -> None:
    """Panel labels, page background, and shared chrome (all pages)."""
    st.markdown(
        """
<style>
/*
 * Full page: white (#fff, same as Plotly plotly_white paper) + subtle dot texture only.
 * Line grid is reserved for the home banner (.ff-hero), not the app shell.
 */
.stApp {
    background-color: #ffffff !important;
    background-image: radial-gradient(rgba(15, 23, 42, 0.055) 1px, transparent 1px) !important;
    background-size: 20px 20px !important;
    background-attachment: fixed !important;
}
[data-testid="stAppViewContainer"] .block-container {
    background-color: transparent !important;
}
[data-testid="stHeader"] {
    background-color: #ffffff !important;
    background-image: radial-gradient(rgba(15, 23, 42, 0.055) 1px, transparent 1px) !important;
    background-size: 20px 20px !important;
    border-bottom: 1px solid rgba(226, 232, 240, 0.95);
    backdrop-filter: none;
}
/* Plotly embed: match page paper colour (avoids grey Streamlit chrome around charts) */
[data-testid="stPlotlyChart"],
[data-testid="stPlotlyChart"] > div,
[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plotly-graph-div {
    background-color: #ffffff !important;
}
/* Sidebar: distinct from main white canvas (theme secondary + light edge) */
[data-testid="stSidebar"] {
    background-color: #f1f5fb !important;
    background-image: radial-gradient(rgba(15, 23, 42, 0.045) 1px, transparent 1px) !important;
    background-size: 18px 18px !important;
    border-right: 1px solid rgba(148, 163, 184, 0.35) !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] p,
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span {
    font-size: 1.05rem !important;
    line-height: 1.35 !important;
}
/* st.title() headings in main column (hero title keeps its own rule below on home) */
section[data-testid="stMain"] h1 {
    font-size: clamp(1.95rem, 3.2vw, 2.35rem) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}
.latent-panel-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #475569;
    margin: 0 0 0.35rem 0;
    letter-spacing: 0.02em;
}
.latent-panel-title-gap { margin-top: 0.85rem; }
</style>
""",
        unsafe_allow_html=True,
    )


def inject_home_landing_styles() -> None:
    """Hero, nav cards, and section labels (home page only)."""
    st.markdown(
        """
<style>
.ff-hero {
    position: relative;
    overflow: hidden;
    border-radius: 16px;
    padding: 1.5rem 1.85rem 1.45rem;
    margin-bottom: 1.35rem;
    border: 1px solid rgba(129, 140, 248, 0.45);
    box-shadow:
        0 4px 24px rgba(30, 27, 75, 0.18),
        inset 0 1px 0 rgba(255, 255, 255, 0.12);
    background:
        linear-gradient(118deg, rgba(15, 23, 42, 0.97) 0%, rgba(49, 46, 129, 0.94) 38%, rgba(67, 56, 202, 0.92) 72%, rgba(79, 70, 229, 0.88) 100%);
}
/* Banner: visible line grid (Shape2Force-style) over the gradient */
.ff-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    opacity: 0.55;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(255, 255, 255, 0.11) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.11) 1px, transparent 1px);
    background-size: 20px 20px;
}
.ff-hero::after {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: radial-gradient(ellipse 85% 65% at 18% 0%, rgba(129, 140, 248, 0.35) 0%, transparent 55%);
}
.ff-hero-inner {
    position: relative;
    z-index: 1;
}
.ff-hero-title-row {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    flex-wrap: wrap;
    margin-bottom: 0.4rem;
}
.ff-hero-emoji {
    font-size: clamp(1.75rem, 4.5vw, 2.35rem);
    line-height: 1;
    filter: drop-shadow(0 2px 8px rgba(0, 0, 0, 0.2));
    user-select: none;
}
/* Narrower hero title so it does not inherit the large st.title size above */
section[data-testid="stMain"] .ff-hero .ff-hero-text h1 {
    font-size: clamp(1.65rem, 4vw, 2.1rem) !important;
    font-weight: 700 !important;
    margin: 0 !important;
    color: #f8fafc !important;
    letter-spacing: -0.03em !important;
    line-height: 1.15 !important;
    text-shadow: 0 1px 18px rgba(0, 0, 0, 0.25) !important;
}
.ff-hero-sub {
    margin: 0;
    max-width: 52rem;
    font-size: 0.98rem;
    line-height: 1.55;
    color: rgba(226, 232, 240, 0.95);
    font-weight: 400;
}
.ff-section-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    margin: 0 0 0.35rem 0;
}
.ff-nav-slot-marker {
    display: block !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    clip-path: inset(50%) !important;
}
/*
 * Home workspace nav cards: each column renders span#ff-nav-slot-N, then the bordered tile.
 * Target the bordered wrapper as the next sibling block after the markdown that contains the span.
 */
section[data-testid="stMain"] *:has(span#ff-nav-slot-1) + * [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stMain"] *:has(span#ff-nav-slot-1) + * + * [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stMain"] *:has(span#ff-nav-slot-2) + * [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stMain"] *:has(span#ff-nav-slot-2) + * + * [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stMain"] *:has(span#ff-nav-slot-3) + * [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stMain"] *:has(span#ff-nav-slot-3) + * + * [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stMain"] *:has(span#ff-nav-slot-4) + * [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stMain"] *:has(span#ff-nav-slot-4) + * + * [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    transition: box-shadow 0.15s ease, border-color 0.15s ease !important;
}
section[data-testid="stMain"] *:has(span#ff-nav-slot-1) + * [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stMain"] *:has(span#ff-nav-slot-1) + * + * [data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(152deg, #eff6ff 0%, #ffffff 52%, #dbeafe 100%) !important;
    border: 1px solid rgba(37, 99, 235, 0.38) !important;
    box-shadow: 0 2px 14px rgba(37, 99, 235, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.92) !important;
}
section[data-testid="stMain"] *:has(span#ff-nav-slot-1) + * [data-testid="stVerticalBlockBorderWrapper"]:hover,
section[data-testid="stMain"] *:has(span#ff-nav-slot-1) + * + * [data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(29, 78, 216, 0.55) !important;
    box-shadow: 0 6px 22px rgba(37, 99, 235, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.95) !important;
}
section[data-testid="stMain"] *:has(span#ff-nav-slot-2) + * [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stMain"] *:has(span#ff-nav-slot-2) + * + * [data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(152deg, #fff7ed 0%, #ffffff 52%, #ffedd5 100%) !important;
    border: 1px solid rgba(234, 88, 12, 0.35) !important;
    box-shadow: 0 2px 14px rgba(234, 88, 12, 0.09), inset 0 1px 0 rgba(255, 255, 255, 0.92) !important;
}
section[data-testid="stMain"] *:has(span#ff-nav-slot-2) + * [data-testid="stVerticalBlockBorderWrapper"]:hover,
section[data-testid="stMain"] *:has(span#ff-nav-slot-2) + * + * [data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(194, 65, 12, 0.5) !important;
    box-shadow: 0 6px 22px rgba(234, 88, 12, 0.14), inset 0 1px 0 rgba(255, 255, 255, 0.95) !important;
}
section[data-testid="stMain"] *:has(span#ff-nav-slot-3) + * [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stMain"] *:has(span#ff-nav-slot-3) + * + * [data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(152deg, #ecfdf5 0%, #ffffff 52%, #d1fae5 100%) !important;
    border: 1px solid rgba(5, 150, 105, 0.36) !important;
    box-shadow: 0 2px 14px rgba(5, 150, 105, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.92) !important;
}
section[data-testid="stMain"] *:has(span#ff-nav-slot-3) + * [data-testid="stVerticalBlockBorderWrapper"]:hover,
section[data-testid="stMain"] *:has(span#ff-nav-slot-3) + * + * [data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(4, 120, 87, 0.52) !important;
    box-shadow: 0 6px 22px rgba(5, 150, 105, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.95) !important;
}
section[data-testid="stMain"] *:has(span#ff-nav-slot-4) + * [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stMain"] *:has(span#ff-nav-slot-4) + * + * [data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(152deg, #f5f3ff 0%, #ffffff 52%, #ede9fe 100%) !important;
    border: 1px solid rgba(124, 58, 237, 0.34) !important;
    box-shadow: 0 2px 14px rgba(124, 58, 237, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.92) !important;
}
section[data-testid="stMain"] *:has(span#ff-nav-slot-4) + * [data-testid="stVerticalBlockBorderWrapper"]:hover,
section[data-testid="stMain"] *:has(span#ff-nav-slot-4) + * + * [data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(109, 40, 217, 0.5) !important;
    box-shadow: 0 6px 22px rgba(124, 58, 237, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.95) !important;
}
</style>
""",
        unsafe_allow_html=True,
    )
