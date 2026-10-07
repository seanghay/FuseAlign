"""FuseAlign: Forced Alignment in the Wild (Vaidya et al., 2026).
https://arxiv.org/abs/2609.33650v1
"""

import json
import math
import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF
from torch import Tensor
from torch.utils.data import Dataset

from fusealign.model import HOP, MS_PER_FRAME, SAMPLE_RATE, WINDOW, encode_text

INSERT_RATE = 0.05


def load_audio(path: str) -> Tensor:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav).mean(1)
    if sr != SAMPLE_RATE:
        wav = AF.resample(wav, sr, SAMPLE_RATE)
    return wav


def augment(wav: Tensor, noises: list[Tensor]) -> Tensor:
    if random.random() > 0.25:
        return wav
    if random.random() < 0.40:
        rt60 = random.uniform(0.1, 1.0)
        drr = random.uniform(0, 25)
        n = int(rt60 * SAMPLE_RATE)
        rir = torch.randn(n) * torch.exp(-6.9 * torch.arange(n) / n)
        rir = rir / rir.norm() * 10 ** (-drr / 20)
        rir[0] = 1.0
        wav = AF.fftconvolve(wav, rir)[: wav.shape[0]]
    if random.random() < 0.40 and noises:
        noise = random.choice(noises)
        off = random.randint(0, max(0, noise.shape[0] - wav.shape[0]))
        noise = noise[off : off + wav.shape[0]]
        noise = torch.nn.functional.pad(noise, (0, wav.shape[0] - noise.shape[0]))
        snr = random.uniform(3, 20)
        wav = AF.add_noise(wav[None], noise[None], torch.tensor([snr]))[0]
    if random.random() < 0.10:
        q = torch.quantile(wav.abs(), 1 - random.uniform(0, 0.10))
        wav = wav.clamp(-q, q)
    if random.random() < 0.30:
        wav = AF.lowpass_biquad(wav, SAMPLE_RATE, random.uniform(2000, 5000))
    if random.random() < 0.15:
        wav = AF.highpass_biquad(wav, SAMPLE_RATE, random.uniform(50, 100))
    if random.random() < 0.40:
        wav = wav * 10 ** (random.uniform(-12, 12) / 20)
    return wav


def plausible_short(words: list[dict], i: int) -> bool:
    w = words[i]
    if len(w["text"]) > 3 or i == 0 or i == len(words) - 1:
        return False
    gap_l = w["start"] - words[i - 1]["end"]
    gap_r = words[i + 1]["start"] - w["end"]
    return gap_l <= 0.010 and gap_r <= 0.010


class AlignDataset(Dataset):
    """Manifest: jsonl lines {"audio": path, "words": [{"text", "start", "end"}, ...]}."""

    def __init__(
        self,
        manifest: str,
        max_sec: float = 30.0,
        train: bool = True,
        noise_dir: str | None = None,
    ):
        self.items = [json.loads(line) for line in open(manifest) if line.strip()]
        self.max_sec, self.train = max_sec, train
        self.noises = []
        if noise_dir:
            paths = sorted(Path(noise_dir).glob("*.wav"))[:512]
            self.noises = [load_audio(str(p))[: 10 * SAMPLE_RATE] for p in paths]

    def __len__(self) -> int:
        return len(self.items)

    def crop(self, wav: Tensor, words: list[dict]) -> tuple[Tensor, list[dict]]:
        dur = wav.shape[0] / SAMPLE_RATE
        if dur <= self.max_sec:
            return wav, words
        i = random.randrange(len(words)) if self.train else 0
        t0 = max(0.0, words[i]["start"] - random.uniform(0, 0.5))
        t1 = min(dur, t0 + self.max_sec)
        kept = [w for w in words if w["start"] >= t0 and w["end"] <= t1]
        if not kept:
            kept = [words[i]]
        t1 = min(t1, kept[-1]["end"] + 0.5) if self.train else t1
        wav = wav[int(t0 * SAMPLE_RATE) : int(t1 * SAMPLE_RATE)]
        return wav, [{**w, "start": w["start"] - t0, "end": w["end"] - t0} for w in kept]

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        wav, words = self.crop(load_audio(item["audio"]), item["words"])
        if self.train:
            wav = augment(wav, self.noises)
        present = [1] * len(words)
        bound = [
            (w["end"] - w["start"]) >= 0.030 or plausible_short(words, i)
            for i, w in enumerate(words)
        ]
        if self.train and len(words) > 1:
            n_ins = sum(random.random() < INSERT_RATE for _ in words)
            for _ in range(n_ins):
                j = random.randrange(len(words) + 1)
                fake = {"text": random.choice(words)["text"], "start": 0.0, "end": 0.0}
                words.insert(j, fake)
                present.insert(j, 0)
                bound.insert(j, False)
        chars, spans = encode_text([w["text"] for w in words])
        return {
            "wav": wav,
            "chars": torch.tensor(chars),
            "spans": spans,
            "start_ms": torch.tensor([round(w["start"] * 1000) for w in words]),
            "end_ms": torch.tensor([round(w["end"] * 1000) for w in words]),
            "present": torch.tensor(present),
            "bound": torch.tensor(bound),
            "words": [w["text"] for w in words],
            "id": item.get("id", str(idx)),
        }


def collate(batch: list[dict]) -> dict:
    b = len(batch)
    S = max(x["wav"].shape[0] for x in batch)
    S = math.ceil(max(S, WINDOW) / HOP) * HOP
    C = max(x["chars"].shape[0] for x in batch)
    W = max(len(x["spans"]) for x in batch)
    wav = torch.zeros(b, S)
    wav_len = torch.zeros(b, dtype=torch.long)
    chars = torch.zeros(b, C, dtype=torch.long)
    pool = torch.zeros(b, W, C)
    start = torch.zeros(b, W, dtype=torch.long)
    end = torch.zeros(b, W, dtype=torch.long)
    present = torch.zeros(b, W, dtype=torch.long)
    word_mask = torch.zeros(b, W, dtype=torch.bool)
    bound = torch.zeros(b, W, dtype=torch.bool)
    for i, x in enumerate(batch):
        n = x["wav"].shape[0]
        wav[i, :n], wav_len[i] = x["wav"], n
        chars[i, : x["chars"].shape[0]] = x["chars"]
        for k, (s, e) in enumerate(x["spans"]):
            pool[i, k, s:e] = 1.0 / (e - s)
        w = len(x["spans"])
        start[i, :w], end[i, :w] = x["start_ms"], x["end_ms"]
        present[i, :w], bound[i, :w] = x["present"], x["bound"]
        word_mask[i, :w] = True
    n_frames = (wav_len - WINDOW) // HOP + 1
    n_ms = ((S - WINDOW) // HOP + 1) * MS_PER_FRAME
    frame_mask = torch.arange(n_ms)[None] < (n_frames * MS_PER_FRAME)[:, None]
    return {
        "wav": wav,
        "wav_len": wav_len,
        "chars": chars,
        "pool": pool,
        "start_ms": start.clamp(max=n_ms - 1),
        "end_ms": end.clamp(max=n_ms - 1),
        "present": present,
        "word_mask": word_mask,
        "bound_mask": bound & word_mask,
        "frame_mask": frame_mask,
        "n_frames": n_frames,
        "words": [x["words"] for x in batch],
        "ids": [x["id"] for x in batch],
    }
