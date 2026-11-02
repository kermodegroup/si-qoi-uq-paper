"""Shared test settings: float64 and the committed model fixture (copied from ace-jax's fixtures/)."""
import os
import pathlib

os.environ.setdefault("JAX_ENABLE_X64", "1")
ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_DIR = pathlib.Path(os.environ.get("SIQ_FIXTURE_DIR", ROOT / "fixtures"))
