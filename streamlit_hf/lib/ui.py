"""Light shared styles (no heavy themes; keeps default Streamlit + plotly_white)."""

from __future__ import annotations

import streamlit as st


def inject_app_styles() -> None:
    """Panel labels and home cards; safe to call on every rerun (small CSS block)."""
    st.markdown(
        """
<style>
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
