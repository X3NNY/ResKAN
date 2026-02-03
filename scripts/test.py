from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import add_src_to_syspath, default_device

repo_root = add_src_to_syspath()

from reskan.checkpoints import CheckpointPaths, load_checkpoint  # noqa: E402
from reskan.data import make_dataloaders  # noqa: E402
from reskan.metrics import calc_ssim, calc_uiq, r2_pcc_scores  # noqa: E402
from reskan.paths import default_data_root, default_results_root  # noqa: E402
from reskan.registry import build_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Test a model checkpoint and report metrics.")
    p.add_argument("--model", type=str, required=True, choices=["CNN", "TCN", "RNN", "RKAN-18", "ResKAN"])
    p.add_argument("--dataset", type=float, default=0, choices=[0, 1, 2], help="0=Marmousi2, 1=Overthrust, 2=Overthrust (interpolated traces)")
    p.add_argument("--phase", type=int, default=0)
    p.add_argument("--freq", type=int, default=30)
    p.add_argument("--snr-db", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default=None, help="Explicit .pkl checkpoint path (overrides default lookup).")
    p.add_argument("--device", type=str, default=default_device())
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--ckpt-dir", type=str, default=None, help="Directory containing checkpoints (default: repo/checkpoints).")
    p.add_argument("--save-plots", action="store_true", default=True)
    return p.parse_args()


def save_result_plots(results_root: Path, name: str, ai: np.ndarray, ai_pred: np.ndarray):
    out_dir = results_root / "img" / "ai"
    out_dir.mkdir(parents=True, exist_ok=True)
    vmin = float(min(ai.min(), ai_pred.min()))
    vmax = float(max(ai.max(), ai_pred.max()))

    def _plot(arr: np.ndarray, title: str, path: Path, cmap: str = "rainbow"):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(arr.T, cmap=cmap, vmin=vmin if cmap != "grey" else None, vmax=vmax if cmap != "grey" else None, aspect="auto")
        ax.set_title(title)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    _plot(ai_pred, f"{name} - Prediction", out_dir / f"{name}_pred.png", cmap="seismic")
    _plot(np.abs(ai_pred - ai), f"{name} - |Diff|", out_dir / f"{name}_diff.png", cmap="seismic")


def main():
    args = parse_args()

    device = args.device
    data_root = Path(args.data_root) if args.data_root else default_data_root()
    results_root = default_results_root()

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (repo_root / "checkpoints")
    ckpt_paths = CheckpointPaths(checkpoints_dir=ckpt_dir)

    bundle = build_model(args.model)
    model = bundle.model

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = ckpt_paths.file_for(bundle.checkpoint_stem, args.phase, args.freq, finetune=False)

    _ = load_checkpoint(ckpt_path, model, device=device)
    print("Loaded checkpoint:", ckpt_path)
    print("Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    data_loader, _ = make_dataloaders(
        data_root=data_root,
        dataset_num=args.dataset,
        phase=args.phase,
        freq=args.freq,
        device=device,
        batch_size=64,
        n_wells=0,
        standardize=True,
        test_only=True,
        test_snr_db=args.snr_db,
    )

    criterion = torch.nn.MSELoss()
    ai_pred = []
    ai = []
    cnt = 0
    loss = 0.0
    inference_times = []

    with torch.no_grad():
        model.eval()
        for x, y, co in data_loader:
            if device == "cuda":
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.time()

            out = model(x)

            if device == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event) / 1000.0
            else:
                elapsed = time.time() - start_time
            inference_times.append(elapsed)

            ai_pred.extend(out.detach().cpu().numpy().squeeze())
            ai.extend(y.detach().cpu().numpy().squeeze())
            loss += float(criterion(out, y).item())
            cnt += 1

    ai_pred = np.array(ai_pred)
    ai = np.array(ai)

    print("Test MSE loss:", loss / max(cnt, 1))
    print("SSIM:", calc_ssim(ai, ai_pred))
    print("r-squared and PCC:", r2_pcc_scores(ai, ai_pred))

    total_time = sum(inference_times)
    print("\nInference Time Statistics:")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Average time per batch: {total_time / max(cnt, 1):.4f} seconds")
    print(f"  Min time: {min(inference_times):.4f} seconds")
    print(f"  Max time: {max(inference_times):.4f} seconds")
    print(f"  Number of batches: {cnt}")

    if args.save_plots:
        save_result_plots(results_root, bundle.paper_name, ai, ai_pred)
        print("Saved plots under", results_root / "img" / "ai")


if __name__ == "__main__":
    main()

