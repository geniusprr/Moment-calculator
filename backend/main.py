"""Vercel entrypoint; reuse the same application and Python solver as local runs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from beam_solver_backend.main import app  # noqa: E402,F401
