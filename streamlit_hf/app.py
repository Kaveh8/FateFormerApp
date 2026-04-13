"""
FateFormer Explorer: interactive analysis hub.
Run from repository root: PYTHONPATH=. streamlit run streamlit_hf/app.py
"""

from pathlib import Path

import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
_ICON_PATH = _APP_DIR / "static" / "app_icon.svg"
_page_icon_kw = {"page_icon": str(_ICON_PATH)} if _ICON_PATH.is_file() else {}

st.set_page_config(
    page_title="FateFormer Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
    **_page_icon_kw,
)

_home = str(_APP_DIR / "home.py")
_p1 = str(_APP_DIR / "pages" / "1_Single_Cell_Explorer.py")
_p2 = str(_APP_DIR / "pages" / "2_Feature_insights.py")
_p3 = str(_APP_DIR / "pages" / "3_Flux_analysis.py")
_p4 = str(_APP_DIR / "pages" / "4_Gene_expression_analysis.py")

pages = [
    st.Page(_home, title="Home", icon=":material/home:", default=True),
    st.Page(_p1, title="Single-Cell Explorer", icon=":material/scatter_plot:"),
    st.Page(_p2, title="Feature Insights", icon=":material/analytics:"),
    st.Page(_p3, title="Flux Analysis", icon=":material/account_tree:"),
    st.Page(_p4, title="Gene Expression & TF Activity", icon=":material/genetics:"),
]
nav = st.navigation(pages)
nav.run()
