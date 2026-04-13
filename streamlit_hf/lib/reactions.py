"""Shared reaction-string normalisation (flux features vs metabolic metadata)."""

from __future__ import annotations

import re


def normalize_reaction_key(name: str) -> str:
    """Map `A→B` style names to the same key as metadata `A -> B` (case-insensitive)."""
    t = str(name).strip().replace("→", " -> ")
    t = re.sub(r"\s+", " ", t)
    return t.lower()
