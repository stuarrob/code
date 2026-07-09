
# --- auto-load .env on package import (cross-platform; replaces WSL `source .env`) ---
from pathlib import Path as _Path
try:
    from dotenv import load_dotenv as _load_dotenv
    _dotenv = _Path(__file__).resolve().parent.parent / ".env"
    if _dotenv.exists():
        _load_dotenv(_dotenv)
except Exception:
    import logging as _logging
    _logging.getLogger(__name__).warning("Could not auto-load .env", exc_info=True)
