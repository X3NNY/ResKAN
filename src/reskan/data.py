from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import segyio
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from .contours import extract_contours


@dataclass(frozen=True)
class DatasetSpec:
    """
    Mirrors your original numeric dataset codes (util/data.py):
    - 0: Marmousi2
    - 2: Overthrust (full)
    - 3: Overthrust (transfer training: interpolated traces)
    """

    dataset_num: float


class SeismicDataset(Dataset):
    def __init__(
        self,
        *,
        split: str,
        seismic: np.ndarray,
        impedance: np.ndarray,
        contour: np.ndarray,
        device: str,
        snr_db: int = 0,
        n_wells: int = 128,
        val_multiplier: int = 10,
        seed: int = 2024,
    ):
        super().__init__()

        rng = np.random.default_rng(seed)
        train_indices = np.linspace(0, impedance.shape[0] - 1, n_wells, dtype=np.int32)

        if split == "train":
            s, ip, co = seismic[train_indices], impedance[train_indices], contour[train_indices]
        elif split == "val":
            # keep close behavior to original util/data.py
            rand_idx = rng.integers(0, impedance.shape[0], size=13601, dtype=np.int64)
            val_indices = np.setdiff1d(rand_idx, train_indices)[: n_wells * val_multiplier]
            s, ip, co = seismic[val_indices], impedance[val_indices], contour[val_indices]
        elif split == "test":
            s, ip, co = seismic, impedance, contour
        else:
            raise ValueError(f"Unknown split: {split}")

        if s.shape[0] == 0:
            return

        # Noise
        if snr_db and snr_db > 0:
            rms = np.sqrt(np.mean(s**2))
            std = rms / (10 ** (snr_db / 20))
            s = s + np.random.normal(0, std, s.shape)

        s = (s - s.mean()) / s.std()
        ip = (ip - ip.mean()) / ip.std()

        self.seismic = torch.from_numpy(np.expand_dims(s, 1)).float().to(device)
        self.impedance = torch.from_numpy(np.expand_dims(ip, 1)).float().to(device)
        self.contour = torch.from_numpy(np.expand_dims(co, 1)).long().to(device)

    def __getitem__(self, idx: int):
        return self.seismic[idx], self.impedance[idx], self.contour[idx]

    def __len__(self) -> int:
        return int(self.seismic.shape[0])


def load_seismic_impedance_contour(
    *,
    data_root: Path,
    dataset_num: float,
    phase: int = 0,
    freq: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load (seismic, impedance, contour) using the same file conventions as the original project.
    """

    if dataset_num == 0:
        seismic_path = data_root / "marmousi2" / f"seismic_{phase}_{freq}.npy"
        if not seismic_path.exists():
            raise FileNotFoundError(f"Seismic file not found: {seismic_path}, please run the forward modeling script (forward_model.py) to generate this file first.")
        seismic = np.load(seismic_path)

        den_file = segyio.open(str(data_root / "marmousi2" / "MODEL_DENSITY_1.25m.segy"))
        rho = segyio.cube(den_file).squeeze()
        v_file = segyio.open(str(data_root / "marmousi2" / "MODEL_P-WAVE_VELOCITY_1.25m.segy"))
        vp = segyio.cube(v_file).squeeze()
        impedance = (vp * rho)[:, :-1]
        contour = extract_contours(impedance)
        return seismic, impedance, contour

    if dataset_num == 1:
        seismic_path = data_root / "overthrust" / f"seismic_{phase}_{freq}.npy"
        if not seismic_path.exists():
            raise FileNotFoundError(f"Seismic file not found: {seismic_path}, please run the forward modeling script (forward_model.py) to generate this file first.")
            raise FileNotFoundError(f"Seismic file not found: {seismic_path}")
        seismic = np.load(seismic_path)
        vp = loadmat(data_root / "overthrust" / "vp.mat")
        d = loadmat(data_root / "overthrust" / "d.mat")
        impedance = (vp["vp"] * d["d"]).T[:, :-1]
        contour = extract_contours(impedance)
        return seismic, impedance, contour
    

    if dataset_num == 2:
        seismic_path = data_root / "overthrust" / f"inter_seismic.npy"
        seismic = np.load(seismic_path)
        impedance_path = data_root / "overthrust" / "inter_impedance.npy"
        impedance = np.load(impedance_path)[:, :-1]
        contour = extract_contours(impedance)
        return seismic, impedance, contour

    raise ValueError(f"Unsupported dataset_num: {dataset_num}")


def make_dataloaders(
    *,
    data_root: Path,
    dataset_num: float,
    phase: int,
    freq: int,
    device: str,
    batch_size: int,
    n_wells: int,
    standardize: bool = True,
    test_only: bool = False,
    test_snr_db: int = 0,
) -> tuple[DataLoader, Optional[DataLoader]]:
    seismic, impedance, contour = load_seismic_impedance_contour(
        data_root=data_root, dataset_num=dataset_num, phase=phase, freq=freq
    )

    if test_only:
        ds = SeismicDataset(
            split="test",
            seismic=seismic,
            impedance=impedance,
            contour=contour,
            device=device,
            snr_db=test_snr_db,
            n_wells=n_wells,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=False), None

    train_ds = SeismicDataset(
        split="train",
        seismic=seismic,
        impedance=impedance,
        contour=contour,
        device=device,
        n_wells=n_wells,
    )
    val_ds = SeismicDataset(
        split="val",
        seismic=seismic,
        impedance=impedance,
        contour=contour,
        device=device,
        n_wells=n_wells,
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )

