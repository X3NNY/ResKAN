from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class CheckpointPaths:
    checkpoints_dir: Path

    def file_for(self, stem: str, phase: int, freq: int, *, finetune: bool = False) -> Path:
        suffix = "_FT" if finetune else ""
        return self.checkpoints_dir / f"{stem}_{phase}_{freq}{suffix}.pkl"


def save_checkpoint(path: Path, model: torch.nn.Module) -> None:
    """
    Save checkpoint in the same pickle format used by the original project:
    {
      \"model\": BytesIO(torch.save(state_dict)),
      \"curves\": {...}
    }
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    payload = {"model": buffer}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path: Path, model: torch.nn.Module, *, device: str) -> Optional[dict[str, Any]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)

    payload["model"].seek(0)
    state_dict = torch.load(payload["model"], weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    return payload.get("curves")

