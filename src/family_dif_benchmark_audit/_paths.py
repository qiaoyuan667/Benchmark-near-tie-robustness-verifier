"""Portable release-root resolution."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("FAMILY_DIF_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
