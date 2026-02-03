from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_syspath() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    sys.path.insert(0, str(src))
    return repo_root


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"

