from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from .models.cnn import CNNModel
from .models.rkan18 import RKAN18
from .models.rnn import RNNInverseModel
from .models.reskan import ResKAN
from .models.tcn import TCNModel

ModelName = Literal["CNN", "TCN", "RNN", "ResKAN", "RKAN-18"]


@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    paper_name: str
    checkpoint_stem: str


def build_model(name: ModelName) -> ModelBundle:
    """
    Construct a model with the default hyperparameters used in this project/paper.
    """
    if name == "CNN":
        m = CNNModel()
        return ModelBundle(model=m, paper_name="CNN", checkpoint_stem="CNN")

    if name == "TCN":
        m = TCNModel(1, 1, [18, 40, 40, 40, 36, 36, 36], 5, 0.3)
        return ModelBundle(model=m, paper_name="TCN", checkpoint_stem="TCN")

    if name == "RNN":
        m = RNNInverseModel(1, factor=3.0)
        return ModelBundle(model=m, paper_name="RNN", checkpoint_stem="RNN")

    if name == "ResKAN":
        m = ResKAN(grid_size=5)
        stem = getattr(m, "checkpoint_name", getattr(m, "name", "ResKAN"))
        return ModelBundle(model=m, paper_name="ResKAN", checkpoint_stem=stem)

    if name == "RKAN-18":
        m = RKAN18(num_classes=1)
        return ModelBundle(model=m, paper_name="RKAN-18", checkpoint_stem="RKAN-18")

    raise ValueError(f"Unknown model: {name}")

