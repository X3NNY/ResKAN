from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from _common import add_src_to_syspath, default_device

repo_root = add_src_to_syspath()

from reskan.checkpoints import CheckpointPaths, load_checkpoint, save_checkpoint  # noqa: E402
from reskan.data import make_dataloaders  # noqa: E402
from reskan.metrics import calc_ssim, calc_uiq, r2_pcc_scores  # noqa: E402
from reskan.paths import default_data_root  # noqa: E402
from reskan.registry import build_model  # noqa: E402
from reskan.models.reskan import ResKANLoss  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Transfer learning: Marmousi2 -> Overthrust")
    p.add_argument("--model", type=str, required=True, choices=["CNN", "TCN", "RNN", "RKAN-18", "ResKAN"])
    p.add_argument("--phase", type=int, default=0)
    p.add_argument("--freq", type=int, default=30)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-wells", type=int, default=61)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-7)
    p.add_argument("--loss-weight-mse", type=float, default=1.0)
    p.add_argument("--loss-weight-ce", type=float, default=0.01)
    p.add_argument("--kan-reg-weight", type=float, default=0.0)
    p.add_argument("--device", type=str, default=default_device())
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--ckpt-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device
    data_root = Path(args.data_root) if args.data_root else default_data_root()
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (repo_root / "checkpoints")
    ckpt_paths = CheckpointPaths(checkpoints_dir=ckpt_dir)

    bundle = build_model(args.model)
    model = bundle.model.to(device)

    # Load Marmousi2 pretrained checkpoint
    pre_ckpt = ckpt_paths.file_for(bundle.checkpoint_stem, args.phase, args.freq, finetune=False)
    _ = load_checkpoint(pre_ckpt, model, device=device)
    print("Loaded pretrained:", pre_ckpt)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = ResKANLoss(loss_weight=(args.loss_weight_mse, args.loss_weight_ce)).to(device)

    # Transfer training set (dataset 3: interpolated overthrust)
    train_loader, _ = make_dataloaders(
        data_root=data_root,
        dataset_num=2,
        phase=args.phase,
        freq=args.freq,
        device=device,
        batch_size=args.batch_size,
        n_wells=args.n_wells,
        standardize=True,
        test_only=False,
        test_snr_db=0,
    )

    train_loss = [np.inf]
    with trange(args.epochs) as t:
        for epoch in t:
            t.set_description(f"epoch: {epoch}")
            last_loss = None
            for x, y, co in train_loader:
                model.train()
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y, co, epoch)
                if args.kan_reg_weight > 0 and hasattr(model, "loss"):
                    loss = loss + args.kan_reg_weight * model.loss()
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())
            train_loss.append(last_loss if last_loss is not None else train_loss[-1])
            t.set_postfix_str(f"train: {train_loss[-1]:.4f}")

    # Evaluate on Overthrust test set (dataset 2)
    test_loader, _ = make_dataloaders(
        data_root=data_root,
        dataset_num=1,
        phase=args.phase,
        freq=args.freq,
        device=device,
        batch_size=64,
        n_wells=0,
        standardize=True,
        test_only=True,
        test_snr_db=0,
    )

    mse = torch.nn.MSELoss()
    ai_pred = []
    ai = []
    cnt = 0
    loss = 0.0
    inference_times = []

    with torch.no_grad():
        model.eval()
        for x, y, co in test_loader:
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
            loss += float(mse(out, y).item())
            cnt += 1

    ai_pred = np.array(ai_pred)
    ai = np.array(ai)

    print("\nOverthrust test metrics after fine-tuning:")
    print("Test MSE loss:", loss / max(cnt, 1))
    print("SSIM:", calc_ssim(ai, ai_pred))
    print("UIQ:", calc_uiq(ai, ai_pred))
    print("r-squared and PCC:", r2_pcc_scores(ai, ai_pred))

    total_time = sum(inference_times)
    print("\nInference Time Statistics:")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Average time per batch: {total_time / max(cnt, 1):.4f} seconds")

    # Save finetuned checkpoint
    ft_ckpt = ckpt_paths.file_for(bundle.checkpoint_stem, args.phase, args.freq, finetune=True)
    save_checkpoint(ft_ckpt, model)
    print("Saved finetuned checkpoint:", ft_ckpt)


if __name__ == "__main__":
    main()

