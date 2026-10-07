"""FuseAlign: Forced Alignment in the Wild (Vaidya et al., 2026).
https://arxiv.org/abs/2609.33650v1
"""

import torch
import torch.nn.functional as F
from torch import Tensor

SIGMA_MS = 10.0
ALPHA, GAMMA = 0.25, 2.0
BETA = 0.01
W_FOCAL, W_ARGMAX, W_PRES = 1.0, 10.0, 0.1


def gaussian_targets(t_star: Tensor, n_ms: int) -> Tensor:
    # t_star [B,W] in ms -> [B,W,n_ms], truncated at 4 sigma
    t = torch.arange(n_ms, device=t_star.device).float()
    d = t[None, None] - t_star[..., None].float()
    g = torch.exp(-(d**2) / (2 * SIGMA_MS**2))
    return g.masked_fill(d.abs() > 4 * SIGMA_MS, 0.0)


def focal_bce(logits: Tensor, g: Tensor, mask: Tensor) -> Tensor:
    # logits, g: [B,W,T]; mask: [B,W,T] valid entries
    p = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, g, reduction="none")
    p_g = p * g + (1 - p) * (1 - g)
    a_g = ALPHA * g + (1 - ALPHA) * (1 - g)
    loss = a_g * (1 - p_g).clamp(min=1e-6) ** GAMMA * bce
    return (loss * mask).sum() / (g * mask).sum().clamp(min=1.0)


def soft_argmax_loss(
    logits: Tensor, t_star: Tensor, frame_mask: Tensor, word_mask: Tensor
) -> Tensor:
    # frame_mask [B,T] valid frames, word_mask [B,W] supervised words
    T = frame_mask.sum(1).float()  # clip length in ms
    z = logits.masked_fill(~frame_mask[:, None], float("-inf"))
    t = torch.arange(logits.shape[-1], device=logits.device).float()
    t_hat = (z.float().softmax(-1) * t).sum(-1)
    x = (t_hat - t_star.float()) / T[:, None]
    loss = F.smooth_l1_loss(x, torch.zeros_like(x), beta=BETA, reduction="none")
    return (loss * word_mask).sum() / word_mask.sum().clamp(min=1.0)


def presence_loss(logits: Tensor, y: Tensor, word_mask: Tensor) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="none")
    pos, neg = word_mask & (y == 1), word_mask & (y == 0)
    terms = [(bce * m).sum() / m.sum() for m in (pos, neg) if m.any()]
    return 0.5 * sum(terms) if terms else logits.sum() * 0


def total_loss(
    hs: Tensor,
    he: Tensor,
    pres: Tensor,
    start_ms: Tensor,
    end_ms: Tensor,
    present: Tensor,
    word_mask: Tensor,
    bound_mask: Tensor,
    frame_mask: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    n_ms = hs.shape[-1]
    m = bound_mask[:, :, None] & frame_mask[:, None, :]
    gs, ge = gaussian_targets(start_ms, n_ms), gaussian_targets(end_ms, n_ms)
    lf = focal_bce(hs.float(), gs, m) + focal_bce(he.float(), ge, m)
    la = soft_argmax_loss(hs, start_ms, frame_mask, bound_mask) + soft_argmax_loss(
        he, end_ms, frame_mask, bound_mask
    )
    lp = presence_loss(pres.float(), present, word_mask)
    loss = W_FOCAL * lf + W_ARGMAX * la + W_PRES * lp
    return loss, {"focal": lf.item(), "argmax": la.item(), "presence": lp.item()}
