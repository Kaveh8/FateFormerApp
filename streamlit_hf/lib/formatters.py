"""Human-readable labels for compact codes used in cached tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Matches interpretation.predictions._get_modality_info letter codes (R/A/F order).
# Short table-friendly labels (no long parentheticals).
_MODALITY_LONG: dict[str, str] = {
    "RAF": "RNA + ATAC + Flux",
    "RA": "RNA + ATAC",
    "RF": "RNA + Flux",
    "AF": "ATAC + Flux",
    "R": "RNA only",
    "A": "ATAC only",
    "F": "Flux only",
    "None": "No modality data",
    "none": "No modality data",
    "nan": "No modality data",
}

# Rename row fields in inspector tables for display.
_FIELD_DISPLAY: dict[str, str] = {
    "label": "CellTag-Multi label",
}

# Latent explorer: table headers and key–value inspector (exclude non-meaningful / internal cols).
LATENT_TABLE_RENAME: dict[str, str] = {
    "label": "CellTag-Multi label",
    "predicted_class": "Predicted fate",
    "predicted_value": "Prediction score",
    "correct": "Prediction correct",
    "pct": "Dominant fate (%)",
    "modality_label": "Available modalities",
    "dataset_idx": "Dataset index",
    "batch_no": "Batch",
    "fold": "CV fold",
    "clone_id": "Clone ID",
    "clone_size": "Clone size",
    "cell_type": "Cell type",
}

LATENT_DROP_FROM_TABLES: frozenset[str] = frozenset({"umap_x", "umap_y", "modality", "pct_decile"})

_NAME_MAP = {**_FIELD_DISPLAY, **LATENT_TABLE_RENAME}


def _format_scalar(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "Yes" if v else "No"
    try:
        if pd.isna(v):
            return ""
    except (ValueError, TypeError):
        pass
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return ""
    return str(v)


def _field_label(name: str, *, fallback_field_display: bool) -> str:
    k = str(name)
    if fallback_field_display:
        return _NAME_MAP.get(k, _FIELD_DISPLAY.get(k, k))
    return _NAME_MAP.get(k, k)


def expand_modality(code) -> str:
    """Map R/A/F codes (e.g. RAF, RA) to full names."""
    if code is None:
        return _MODALITY_LONG["None"]
    try:
        if pd.isna(code):
            return _MODALITY_LONG["None"]
    except (ValueError, TypeError):
        pass
    if isinstance(code, (float, np.floating)) and np.isnan(code):
        return _MODALITY_LONG["None"]
    key = str(code).strip()
    if not key or key.lower() == "nan":
        return _MODALITY_LONG["None"]
    return _MODALITY_LONG.get(key, key)


def annotate_modality_column(df, code_col: str = "modality", label_col: str = "modality_label"):
    """Add human-readable modality column; returns a copy."""
    out = df.copy()
    out[label_col] = out[code_col].map(expand_modality)
    return out


def prepare_latent_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop UMAP / internal columns and rename headers for Selected-points style tables."""
    drop = [c for c in df.columns if c in LATENT_DROP_FROM_TABLES or str(c).startswith("umap_")]
    out = df.drop(columns=drop, errors="ignore")
    return out.rename(columns=LATENT_TABLE_RENAME)


def latent_inspector_key_value(series: pd.Series) -> pd.DataFrame:
    """Key–value inspector row: human names, no UMAP coordinates."""
    s = series.drop(
        labels=[c for c in series.index if c in LATENT_DROP_FROM_TABLES or str(c).startswith("umap_")],
        errors="ignore",
    )
    idx = [_field_label(i, fallback_field_display=False) for i in s.index]
    vals = [_format_scalar(v) for v in s.values]
    return pd.DataFrame({"Field": idx, "Value": vals})


def dataframe_to_arrow_safe_kv(series: pd.Series) -> pd.DataFrame:
    """Two string columns for Streamlit/PyArrow (avoids mixed-type single column)."""
    s = series.copy()
    idx = [_field_label(i, fallback_field_display=True) for i in s.index]
    vals = [_format_scalar(v) for v in s.values]
    return pd.DataFrame({"field": idx, "value": vals})
