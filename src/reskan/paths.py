from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    # repo/src/reskan/paths.py -> repo/
    return Path(__file__).parent.parent.parent


def default_data_root() -> Path:
    """
    Prefer `repo/data/` (clean repo), otherwise fall back to `../data/`
    (your current project layout where `repo/` is a subfolder).
    """
    r = repo_root()
    return r / "data"


def default_results_root() -> Path:
    r = repo_root()
    return r / "results"

