"""FuseAlign: Forced Alignment in the Wild (Vaidya et al., 2026).
https://arxiv.org/abs/2609.33650v1
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SAMPLE_RATE = 16000
WINDOW = 512
HOP = 256
UPSAMPLE = 4  # 16 ms -> 1 ms
MS_PER_FRAME = HOP * 1000 // SAMPLE_RATE  # 16

VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789 '"
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(VOCAB)}  # 0 = pad / unk


def encode_text(words: list[str]) -> tuple[list[int], list[tuple[int, int]]]:
    ids, spans = [], []
    for i, w in enumerate(words):
        if i:
            ids.append(CHAR_TO_ID[" "])
        start = len(ids)
        ids.extend(CHAR_TO_ID.get(c, 0) for c in w.lower())
        if len(ids) == start:
            ids.append(0)
        spans.append((start, len(ids)))
    return ids, spans


@dataclass
class Config:
    dim: int = 512
    heads: int = 8
    layers: int = 12
    ffn: int = 2048
    head_dim: int = 256
    max_audio_sec: float = 30.0


def rope(x: Tensor, pos: Tensor) -> Tensor:
    # x: [B, H, N, D], pos: [B, N]
    d = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device).float() / d))
    ang = pos[:, None, :, None].float() * inv  # [B,1,N,D/2]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.heads = cfg.heads
        self.n1 = nn.LayerNorm(cfg.dim)
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.n2 = nn.LayerNorm(cfg.dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.dim, cfg.ffn), nn.GELU(), nn.Linear(cfg.ffn, cfg.dim)
        )

    def forward(self, x: Tensor, pos: Tensor, mask: Tensor) -> Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos), rope(k, pos)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.ffn(self.n2(x))


class Upsampler(nn.Module):
    def __init__(self, dim: int, out: int, ch: int = 256):
        super().__init__()
        layers = [nn.Conv1d(dim, ch, 1)]
        for _ in range(UPSAMPLE):
            layers += [nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1), nn.GELU()]
        layers += [nn.Conv1d(ch, ch, 5, padding=2), nn.GELU(), nn.Conv1d(ch, out, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # [B, L, dim] -> [B, 16L, out]
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class FuseAlign(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.audio_in = nn.Conv1d(1, cfg.dim, WINDOW, stride=HOP)
        self.char_emb = nn.Embedding(len(VOCAB) + 1, cfg.dim)
        self.modality = nn.Embedding(2, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.dim)
        self.up_s = Upsampler(cfg.dim, cfg.head_dim)
        self.up_e = Upsampler(cfg.dim, cfg.head_dim)
        self.word_s = nn.Linear(cfg.dim, cfg.head_dim)
        self.word_e = nn.Linear(cfg.dim, cfg.head_dim)
        self.presence = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, 1))
        nn.init.constant_(self.presence[-1].bias, 2.0)

    def forward(
        self, wav: Tensor, wav_len: Tensor, chars: Tensor, pool: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """wav [B,S], wav_len [B], chars [B,C], pool [B,W,C] (row-normalized word pooling).

        Returns start/end heatmap logits [B,W,16L] and presence logits [B,W]."""
        b = wav.shape[0]
        a = self.audio_in(wav[:, None]).transpose(1, 2)  # [B,L,D]
        n_frames = ((wav_len - WINDOW) // HOP + 1).clamp(min=1)
        L, C = a.shape[1], chars.shape[1]
        a = a + self.modality.weight[0]
        t = self.char_emb(chars) + self.modality.weight[1]

        a_mask = torch.arange(L, device=wav.device)[None] < n_frames[:, None]
        t_mask = chars > 0
        a_pos = torch.arange(L, device=wav.device)[None].expand(b, L).float()
        c_len = t_mask.sum(1).clamp(min=2)
        t_pos = torch.arange(C, device=wav.device)[None] / (c_len[:, None] - 1) * n_frames[:, None]

        x = torch.cat([a, t], 1)
        pos = torch.cat([a_pos, t_pos], 1)
        mask = torch.cat([a_mask, t_mask], 1)
        for blk in self.blocks:
            x = blk(x, pos, mask)
        x = self.norm(x)
        a, t = x[:, :L], x[:, L:]

        words = torch.bmm(pool, t)  # [B,W,D]
        ys, ye = self.up_s(a), self.up_e(a)  # [B,16L,H]
        hs = torch.bmm(self.word_s(words), ys.transpose(1, 2))
        he = torch.bmm(self.word_e(words), ye.transpose(1, 2))
        return hs, he, self.presence(words).squeeze(-1)


@torch.no_grad()
def decode(
    hs: Tensor, he: Tensor, pres: Tensor, n_frames: Tensor, p_min: float = 0.1
) -> tuple[Tensor, Tensor, Tensor]:
    """Peak-pick boundaries in ms. Returns start_ms, end_ms [B,W] and found [B,W]."""
    valid = torch.arange(hs.shape[-1], device=hs.device)[None] < (n_frames * MS_PER_FRAME)[:, None]
    hs = hs.masked_fill(~valid[:, None], float("-inf"))
    he = he.masked_fill(~valid[:, None], float("-inf"))
    return hs.argmax(-1), he.argmax(-1), pres.sigmoid() >= p_min
