from __future__ import annotations

from pathlib import Path

import numpy as np
import segyio
from scipy.io import loadmat


def ricker_wavelet(f_hz: float, dt: float, phase_deg: float = 0.0) -> np.ndarray:
    """
    Ricker wavelet with optional phase rotation (degrees).
    """
    nw = 2.2 / (f_hz * dt)
    nw = 2 * np.floor(nw / 2) + 1

    phase = np.deg2rad(phase_deg)
    k = np.arange(nw) + 1
    alpha = (nw // 2 - k + 1) * f_hz * dt * np.pi
    beta = alpha**2
    y = (1 - 2 * beta) * np.exp(-beta)

    # phase rotation (simple approximation used in your original script)
    y = y * np.cos(phase) - np.roll(y, 1) * np.sin(phase)
    return y


def reflectivity_from_impedance(ai: np.ndarray) -> np.ndarray:
    """
    ai: (depth, trace) or (n_samples, n_traces)
    returns r: (depth-1, trace)
    """
    v1 = ai[:-1, :]
    v2 = ai[1:, :]
    return (v2 - v1) / (v2 + v1 + 1e-12)


def load_marmousi2_ai(data_root: Path) -> np.ndarray:
    den_file = segyio.open(str(data_root / "marmousi2" / "MODEL_DENSITY_1.25m.segy"))
    rho = segyio.cube(den_file).squeeze().T
    v_file = segyio.open(str(data_root / "marmousi2" / "MODEL_P-WAVE_VELOCITY_1.25m.segy"))
    vp = segyio.cube(v_file).squeeze().T
    return vp * rho


def load_overthrust_ai(data_root: Path) -> np.ndarray:
    vp = loadmat(data_root / "overthrust" / "vp.mat")["vp"]
    d = loadmat(data_root / "overthrust" / "d.mat")["d"]
    return (vp * d)


def forward_seismic_from_ai(
    ai: np.ndarray,
    *,
    freq_hz: float,
    dt: float,
    phase_deg: float,
) -> np.ndarray:
    """
    Generate seismic traces by convolving reflectivity with a Ricker wavelet.
    Returns traces shaped (n_traces, n_samples_minus_1).
    """
    w = ricker_wavelet(freq_hz, dt, phase_deg)
    rmodel = reflectivity_from_impedance(ai).T  # (trace, depth-1)
    traces = np.zeros_like(rmodel)
    for i in range(rmodel.shape[0]):
        traces[i] = np.convolve(w, rmodel[i, :], mode="same")
    return traces

