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
_fi = _APP_DIR / "pages" / "feature_insights"
_flux = _APP_DIR / "pages" / "flux_analysis"
_ge = _APP_DIR / "pages" / "gene_expression"

pages = {
    "": [
        st.Page(_home, title="Home", icon=":material/home:", default=True),
        st.Page(_p1, title="Single-Cell Explorer", icon=":material/scatter_plot:"),
    ],
    "Feature Insights": [
        st.Page(str(_fi / "1_Global_overview.py"), title="Global overview", icon=":material/dashboard:"),
        st.Page(str(_fi / "2_Modality_spotlight.py"), title="Modality spotlight", icon=":material/view_column:"),
        st.Page(str(_fi / "3_Shift_vs_attention.py"), title="Shift vs attention", icon=":material/scatter_plot:"),
        st.Page(str(_fi / "4_Attention_vs_prediction.py"), title="Attention vs prediction", icon=":material/psychology:"),
        st.Page(str(_fi / "5_Full_table.py"), title="Full table", icon=":material/table:"),
    ],
    "Flux Analysis": [
        st.Page(str(_flux / "5_Interactive_map.py"), title="Metabolic map", icon=":material/map:"),
        st.Page(str(_flux / "1_Pathway_map.py"), title="Pathway map", icon=":material/hub:"),
        st.Page(str(_flux / "2_Differential_fate.py"), title="Differential & fate", icon=":material/compare_arrows:"),
        st.Page(str(_flux / "3_Reaction_ranking.py"), title="Reaction ranking", icon=":material/format_list_numbered:"),
        st.Page(str(_flux / "4_Model_metadata.py"), title="Model metadata", icon=":material/schema:"),
    ],
    "Gene Expression & TF": [
        st.Page(str(_ge / "1_Pathway_enrichment.py"), title="Pathway enrichment", icon=":material/bubble_chart:"),
        st.Page(str(_ge / "2_Motif_activity.py"), title="Motif activity", icon=":material/biotech:"),
        st.Page(str(_ge / "3_Gene_table.py"), title="Gene table", icon=":material/table_rows:"),
        st.Page(str(_ge / "4_Motif_table.py"), title="Motif table", icon=":material/table_chart:"),
    ],
}
nav = st.navigation(pages)
nav.run()
