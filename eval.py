"""Evaluate a FuseAlign checkpoint with the AlignBench protocol (asymmetric, worst-edge, all-word).

Paper: FuseAlign: Forced Alignment in the Wild, https://arxiv.org/abs/2609.33650v1
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fusealign.data import AlignDataset, collate
from fusealign.metrics import score
from fusealign.model import SAMPLE_RATE, Config, FuseAlign, decode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--weights", choices=["model", "ema"], default="ema")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--p-min", type=float, default=0.1)
    p.add_argument("--out", help="write per-word predictions as jsonl")
    p.add_argument("--bf16", action="store_true")
    args = p.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    ck = torch.load(args.ckpt, map_location=device)
    model = FuseAlign(Config(**ck["cfg"])).to(device).eval()
    model.load_state_dict(ck[args.weights])
    amp = {"device_type": device.split(":")[0], "dtype": torch.bfloat16, "enabled": args.bf16}

    ds = AlignDataset(args.manifest, train=False)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=4, collate_fn=collate)
    clips, rows = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.autocast(**amp):
                hs, he, pres = model(batch["wav"], batch["wav_len"], batch["chars"], batch["pool"])
            s, e, found = decode(hs, he, pres, batch["n_frames"], args.p_min)
            for i, cid in enumerate(batch["ids"]):
                n = batch["word_mask"][i].sum().item()
                clip = {
                    "pred_start": s[i, :n].cpu().numpy() / 1000,
                    "pred_end": e[i, :n].cpu().numpy() / 1000,
                    "found": found[i, :n].cpu().numpy(),
                    "ref_start": batch["start_ms"][i, :n].cpu().numpy() / 1000,
                    "ref_end": batch["end_ms"][i, :n].cpu().numpy() / 1000,
                    "dur": batch["wav_len"][i].item() / SAMPLE_RATE,
                }
                clips.append(clip)
                rows.append(
                    {
                        "id": cid,
                        "words": [
                            {"text": w, "start": float(ps), "end": float(pe), "found": bool(f)}
                            for w, ps, pe, f in zip(
                                batch["words"][i],
                                clip["pred_start"],
                                clip["pred_end"],
                                clip["found"],
                                strict=True,
                            )
                        ],
                    }
                )
    for k, v in score(clips).items():
        print(f"{k:>16}: {v:.2f}")
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
