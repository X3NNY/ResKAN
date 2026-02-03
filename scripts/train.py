from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from _common import add_src_to_syspath, default_device

repo_root = add_src_to_syspath()

from reskan.checkpoints import CheckpointPaths, save_checkpoint  # noqa: E402
from reskan.data import make_dataloaders  # noqa: E402
from reskan.paths import default_data_root, default_results_root  # noqa: E402
from reskan.registry import build_model  # noqa: E402
from reskan.models.reskan import ResKANLoss  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train a model (CNN/TCN/RNN/RKAN-18/ResKAN)")
    p.add_argument("--model", type=str, required=True, choices=["CNN", "TCN", "RNN", "RKAN-18", "ResKAN"])
    p.add_argument("--dataset", type=float, default=0, choices=[0, 1, 2], help="0=Marmousi2, 1=Overthrust, 2=Overthrust (interpolated traces)")
    p.add_argument("--phase", type=int, default=0)
    p.add_argument("--freq", type=int, default=30)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-wells", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss-weight-mse", type=float, default=1.0)
    p.add_argument("--loss-weight-ce", type=float, default=0.0)
    p.add_argument("--device", type=str, default=default_device())
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None, help="Checkpoints output dir (default: repo/checkpoints)")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device
    data_root = Path(args.data_root) if args.data_root else default_data_root()
    results_root = default_results_root()
    ckpt_dir = Path(args.out_dir) if args.out_dir else (repo_root / "checkpoints")
    ckpt_paths = CheckpointPaths(checkpoints_dir=ckpt_dir)

    bundle = build_model(args.model)
    model = bundle.model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = ResKANLoss(loss_weight=(args.loss_weight_mse, args.loss_weight_ce)).to(device)

    train_loader, val_loader = make_dataloaders(
        data_root=data_root,
        dataset_num=args.dataset,
        phase=args.phase,
        freq=args.freq,
        device=device,
        batch_size=args.batch_size,
        n_wells=args.n_wells,
        standardize=True,
        test_only=False,
        test_snr_db=0,
    )

    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model:", bundle.paper_name)
    print("Parameters:", params_count)

    train_loss = [np.inf]
    val_loss = [np.inf]

    with trange(args.epochs) as t:
        for epoch in t:
            t.set_description(f"epoch: {epoch}")
            cnt = 0
            tloss = 0.0
            for x, y, co in train_loader:
                model.train()
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y, co, epoch)

                loss.backward()
                optimizer.step()
                tloss += float(loss.item())
                cnt += 1

            train_loss.append(tloss / max(cnt, 1))

            # Validation every 20 epochs (mirrors original)
            if epoch % 20 == 0 and val_loader is not None:
                vcnt = 0
                vloss = 0.0
                with torch.no_grad():
                    model.eval()
                    for x, y, co in val_loader:
                        out = model(x)
                        vloss += float(criterion(out, y, co, epoch).item())
                        vcnt += 1
                val_loss.append(vloss / max(vcnt, 1))
            else:
                val_loss.append(val_loss[-1])

            t.set_postfix_str(f"train: {train_loss[-1]:.4f} | val: {val_loss[-1]:.4f}")

    # Save checkpoint
    stem = bundle.checkpoint_stem
    ckpt_path = ckpt_paths.file_for(stem, args.phase, args.freq, finetune=False)
    save_checkpoint(ckpt_path, model)
    print("Saved checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()

