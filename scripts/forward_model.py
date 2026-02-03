from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import add_src_to_syspath

repo_root = add_src_to_syspath()

from reskan.forward import forward_seismic_from_ai, load_marmousi2_ai, load_overthrust_ai  # noqa: E402
from reskan.paths import default_data_root  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Forward modeling: impedance -> reflectivity -> seismic")
    p.add_argument("--model", type=str, default="marmousi2", choices=["marmousi2", "overthrust"])
    p.add_argument("--phase", type=float, default=0.0)
    p.add_argument("--freq", type=float, default=30.0)
    p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--output", type=str, default=None, help="Output .npy path (default matches original naming).")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root) if args.data_root else default_data_root()

    if args.model == "marmousi2":
        ai = load_marmousi2_ai(data_root)
        default_out = data_root / "marmousi2" / f"seismic_{int(args.phase)}_{int(args.freq)}.npy"
    else:
        ai = load_overthrust_ai(data_root)
        default_out = data_root / "overthrust" / f"seismic_{int(args.phase)}_{int(args.freq)}.npy"

    traces = forward_seismic_from_ai(ai, freq_hz=args.freq, dt=args.dt, phase_deg=args.phase)
    print(traces.shape)
    out_path = Path(args.output) if args.output else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, traces)
    print("Saved:", out_path)
    print("Shape:", traces.shape)


if __name__ == "__main__":
    main()

