"""Train FuseAlign with EMA-snapshot label correction.

Paper: FuseAlign: Forced Alignment in the Wild, https://arxiv.org/abs/2609.33650v1
"""

import argparse
import copy
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fusealign.data import AlignDataset, collate
from fusealign.loss import total_loss
from fusealign.model import Config, FuseAlign, decode

KEEP_MS, AGREE_MS = 50, 20


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val")
    p.add_argument("--noise-dir")
    p.add_argument("--out", default="runs/fusealign")
    p.add_argument("--resume")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--wd", type=float, default=0.03)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--ema-start", type=int, default=20_000)
    p.add_argument("--ema-every", type=int, default=10_000)
    p.add_argument("--ema-bank", type=int, default=3)
    p.add_argument("--no-ema-correction", action="store_true")
    p.add_argument("--ckpt-every", type=int, default=10_000)
    p.add_argument("--val-every", type=int, default=5000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--ffn", type=int, default=2048)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def lr_at(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup:
        return args.lr * step / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def param_groups(model: torch.nn.Module, wd: float) -> list[dict]:
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "emb" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "emb" not in n)]
    return [{"params": decay, "weight_decay": wd}, {"params": other, "weight_decay": 0.0}]


@torch.no_grad()
def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for pe, pm in zip(ema.parameters(), model.parameters(), strict=True):
        pe.lerp_(pm, 1 - decay)


@torch.no_grad()
def correct_labels(bank: list[torch.nn.Module], batch: dict, full: bool, amp: dict) -> dict:
    """Keep Gentle label if all teachers agree with it, replace by teacher median if a full
    bank agrees internally, otherwise mask the boundary loss."""
    preds = []
    for t in bank:
        with torch.autocast(**amp):
            hs, he, pres = t(batch["wav"], batch["wav_len"], batch["chars"], batch["pool"])
        s, e, _ = decode(hs, he, pres, batch["n_frames"])
        preds.append(torch.stack([s, e]))
    p = torch.stack(preds).float()  # [K,2,B,W]
    label = torch.stack([batch["start_ms"], batch["end_ms"]]).float()
    keep = ((p - label).abs() <= KEEP_MS).all(0).all(0)
    agree = full & ((p.amax(0) - p.amin(0)) <= AGREE_MS).all(0)
    median = p.median(0).values
    new = torch.where(keep[None], label, median).long()
    batch["start_ms"], batch["end_ms"] = new[0], new[1]
    batch["bound_mask"] = batch["bound_mask"] & (keep | agree)
    return batch


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: str, amp: dict) -> dict:
    from fusealign.metrics import score

    model.eval()
    clips = []
    for batch in tqdm(loader, desc="val", leave=False):
        batch = to_device(batch, device)
        with torch.autocast(**amp):
            hs, he, pres = model(batch["wav"], batch["wav_len"], batch["chars"], batch["pool"])
        s, e, found = decode(hs, he, pres, batch["n_frames"])
        for i in range(len(batch["ids"])):
            n = batch["word_mask"][i].sum().item()
            clips.append(
                {
                    "pred_start": s[i, :n].cpu().numpy() / 1000,
                    "pred_end": e[i, :n].cpu().numpy() / 1000,
                    "found": found[i, :n].cpu().numpy(),
                    "ref_start": batch["start_ms"][i, :n].cpu().numpy() / 1000,
                    "ref_end": batch["end_ms"][i, :n].cpu().numpy() / 1000,
                    "dur": batch["wav_len"][i].item() / 16000,
                }
            )
    model.train()
    return score(clips)


def to_device(batch: dict, device: str) -> dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()
    }


def save(path: Path, model, ema, bank, opt, step: int, cfg: Config) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "bank": [t.state_dict() for t in bank],
            "opt": opt.state_dict(),
            "step": step,
            "cfg": cfg.__dict__,
        },
        path,
    )


def main() -> None:
    args = parse()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    amp = {"device_type": device.split(":")[0], "dtype": torch.bfloat16, "enabled": args.bf16}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out)

    cfg = Config(dim=args.dim, heads=args.heads, layers=args.layers, ffn=args.ffn)
    model = FuseAlign(cfg).to(device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    bank: list[torch.nn.Module] = []
    opt = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        for sd in ck["bank"]:
            t = copy.deepcopy(ema)
            t.load_state_dict(sd)
            bank.append(t)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, device: {device}")

    train_ds = AlignDataset(args.train, cfg.max_audio_sec, train=True, noise_dir=args.noise_dir)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=device == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = None
    if args.val:
        val_ds = AlignDataset(args.val, cfg.max_audio_sec, train=False)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate
        )

    model.train()
    pbar = tqdm(total=args.steps, initial=step, dynamic_ncols=True)
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            batch = to_device(batch, device)
            if bank and not args.no_ema_correction:
                batch = correct_labels(bank, batch, len(bank) == args.ema_bank, amp)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args)
            with torch.autocast(**amp):
                hs, he, pres = model(batch["wav"], batch["wav_len"], batch["chars"], batch["pool"])
            loss, parts = total_loss(
                hs,
                he,
                pres,
                batch["start_ms"],
                batch["end_ms"],
                batch["present"],
                batch["word_mask"],
                batch["bound_mask"],
                batch["frame_mask"],
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_update(ema, model, args.ema_decay)
            step += 1

            if step >= args.ema_start and (step - args.ema_start) % args.ema_every == 0:
                bank.append(copy.deepcopy(ema))
                bank = bank[-args.ema_bank :]

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.3f}", **{k: f"{v:.3f}" for k, v in parts.items()})
            if step % 20 == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                for k, v in parts.items():
                    writer.add_scalar(f"train/{k}", v, step)
                writer.add_scalar("train/lr", lr_at(step, args), step)
                writer.add_scalar("train/grad_norm", gn.item(), step)
                writer.add_scalar(
                    "train/masked_frac", 1 - batch["bound_mask"].float().mean().item(), step
                )
            if val_loader and step % args.val_every == 0:
                for name, m in (("model", model), ("ema", ema)):
                    for k, v in validate(m, val_loader, device, amp).items():
                        writer.add_scalar(f"val_{name}/{k}", v, step)
            if step % args.ckpt_every == 0:
                save(out / f"step_{step}.pt", model, ema, bank, opt, step, cfg)
                save(out / "last.pt", model, ema, bank, opt, step, cfg)
    save(out / "last.pt", model, ema, bank, opt, step, cfg)
    writer.close()


if __name__ == "__main__":
    main()
