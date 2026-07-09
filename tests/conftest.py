"""Shared pytest configuration for the ETFTrader test suite.

The execution helper scripts under `notebooks/scripts/` (s1_-s12_) are not
installed as part of the `src.*` package, so tests that need to import them
require the directory on `sys.path`.
"""
import sys
from pathlib import Path

_NOTEBOOK_SCRIPTS = Path(__file__).resolve().parent.parent / "notebooks" / "scripts"
if str(_NOTEBOOK_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_SCRIPTS))
