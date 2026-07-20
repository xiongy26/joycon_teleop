#!/usr/bin/env python3
"""Train a compact ACT-style policy on collected EL-A3 HDF5 episodes."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _load_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for ACT training. Install with: pip install torch"
        ) from exc
    return torch, DataLoader, Subset


def _make_loaders(args):
    torch, DataLoader, Subset = _load_torch()
    from act_data.dataset import ACTHDF5Dataset

    camera_names = args.cameras.split(",") if args.cameras else None
    dataset = ACTHDF5Dataset(
        args.dataset,
        chunk_size=args.chunk_size,
        camera_names=camera_names,
        image_downsample=args.image_downsample,
        seed=args.seed,
    )
    train_indices, val_indices = dataset.train_val_split(args.val_ratio)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )
    val_loader = None
    if val_indices:
        val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            drop_last=False,
        )
    return dataset, train_loader, val_loader


def _infer_dims(batch: dict) -> tuple[int, int, int]:
    qpos_dim = int(batch["qpos"].shape[-1])
    action_dim = int(batch["action"].shape[-1])
    num_cameras = int(batch["image"].shape[1])
    return qpos_dim, action_dim, num_cameras


def _run_epoch(model, loader, optimizer, device, *, train: bool, grad_clip: float) -> float:
    torch, _, _ = _load_torch()
    from act_data.policy import masked_l1_loss

    model.train(train)
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        image = batch["image"].to(device=device, non_blocking=True)
        qpos = batch["qpos"].to(device=device, non_blocking=True)
        action = batch["action"].to(device=device, non_blocking=True)
        is_pad = batch["is_pad"].to(device=device, non_blocking=True)

        with torch.set_grad_enabled(train):
            pred = model(image, qpos)
            loss = masked_l1_loss(pred, action, is_pad)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def _save_checkpoint(path: Path, model, optimizer, args, *, epoch: int, train_loss: float, val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch, _, _ = _load_torch()
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": vars(args),
        "model_config": getattr(model, "export_config", lambda: {})(),
    }
    torch.save(payload, path)


def train(args: argparse.Namespace) -> None:
    torch, _, _ = _load_torch()
    from act_data.policy import ACTPolicy

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")

    dataset, train_loader, val_loader = _make_loaders(args)
    first_batch = next(iter(train_loader))
    qpos_dim, action_dim, num_cameras = _infer_dims(first_batch)

    model = ACTPolicy(
        qpos_dim=qpos_dim,
        action_dim=action_dim,
        num_cameras=num_cameras,
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "train_config.json"
    config = {
        **vars(args),
        "qpos_dim": qpos_dim,
        "action_dim": action_dim,
        "num_cameras": num_cameras,
        "camera_names": dataset.camera_names,
        "num_episodes": len(dataset.episode_paths),
        "num_samples": len(dataset),
    }
    model_config = {
        "qpos_dim": qpos_dim,
        "action_dim": action_dim,
        "num_cameras": num_cameras,
        "chunk_size": args.chunk_size,
        "hidden_dim": args.hidden_dim,
        "nheads": args.nheads,
        "num_encoder_layers": args.encoder_layers,
        "num_decoder_layers": args.decoder_layers,
        "dropout": args.dropout,
        "camera_names": dataset.camera_names,
        "image_downsample": args.image_downsample,
    }
    config["model_config"] = model_config
    model.export_config = lambda: model_config
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    best_val = math.inf
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            grad_clip=args.grad_clip,
        )
        val_loss = None
        if val_loader is not None:
            val_loss = _run_epoch(
                model,
                val_loader,
                optimizer,
                device,
                train=False,
                grad_clip=0.0,
            )

        metric = train_loss if val_loss is None else val_loss
        if metric < best_val:
            best_val = metric
            _save_checkpoint(
                output_dir / "policy_best.pt",
                model,
                optimizer,
                args,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )
        if epoch % args.save_every == 0 or epoch == args.epochs:
            _save_checkpoint(
                output_dir / f"policy_epoch_{epoch:04d}.pt",
                model,
                optimizer,
                args,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )

        val_text = "n/a" if val_loss is None else f"{val_loss:.6f}"
        elapsed = time.time() - start
        print(
            f"epoch {epoch:04d}/{args.epochs} "
            f"train_l1={train_loss:.6f} val_l1={val_text} elapsed={elapsed:.1f}s",
            flush=True,
        )

    print(f"saved checkpoints to {output_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="HDF5 episode file or directory. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", default="checkpoints/act_policy")
    parser.add_argument("--cameras", default=None, help="Comma-separated camera names")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--image-downsample", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    args = parser.parse_args(argv)

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.save_every <= 0:
        parser.error("--save-every must be positive")

    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
