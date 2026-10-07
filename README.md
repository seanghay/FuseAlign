# FuseAlign (PyTorch)

Unofficial reimplementation of **FuseAlign: Forced Alignment in the Wild**
(Vaidya, Bailey, Badam, Bendel, He; Descript, 2026). https://arxiv.org/abs/2609.33650v1

Joint audio-text transformer on 16 ms audio patches, convolutional upsampling to 1 ms
boundary heatmaps, a presence head for absent words, focal + soft-argmax losses, and
online EMA-snapshot label correction (paper Sec. 5, App. F).

Differences from the paper: learned character embeddings replace the frozen Dia text
encoder (ablation 5, within noise), default size is the ~60M config, and long-form
chunk routing (App. F.3) is not included, so clips must be <= 30 s.

## Setup

```sh
uv sync
uv run ruff check .
```

## Data

A jsonl manifest, one clip per line, times in seconds:

```json
{"id": "clip1", "audio": "/path/clip1.wav", "words": [{"text": "hello", "start": 0.12, "end": 0.41}, ...]}
```

## Train

```sh
uv run train.py --train train.jsonl --val dev.jsonl --noise-dir noises/ --out runs/base --bf16
uv run train.py --resume runs/base/last.pt --train train.jsonl --val dev.jsonl --out runs/base --bf16
uv run tensorboard --logdir runs
```

Checkpoints (`step_N.pt`, `last.pt`) hold model, EMA, snapshot bank, optimizer, and step.

## Eval

```sh
uv run eval.py --ckpt runs/base/last.pt --manifest alignbench.jsonl --out preds.jsonl
```

Reports mean asymmetric / symmetric error, worst-edge error rates at 25/50/300 ms with the
uniform fallback for unplaced words, and detection F1 (paper Sec. 3).
