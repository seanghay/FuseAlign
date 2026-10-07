"""FuseAlign: Forced Alignment in the Wild (Vaidya et al., 2026).
https://arxiv.org/abs/2609.33650v1
"""

import numpy as np

TOLERANCES_MS = (25, 50, 300)


def fill_missing(start: np.ndarray, end: np.ndarray, found: np.ndarray, dur: float) -> tuple:
    """Uniformly divide the gap between placed neighbours among runs of unplaced words."""
    start, end = start.astype(float).copy(), end.astype(float).copy()
    n, i = len(start), 0
    while i < n:
        if found[i]:
            i += 1
            continue
        j = i
        while j < n and not found[j]:
            j += 1
        lo = end[i - 1] if i > 0 else 0.0
        hi = start[j] if j < n else dur
        edges = np.linspace(lo, max(lo, hi), j - i + 1)
        start[i:j], end[i:j] = edges[:-1], edges[1:]
        i = j
    return start, end


def asymmetric_errors(ps: np.ndarray, pe: np.ndarray, rs: np.ndarray, re: np.ndarray) -> tuple:
    prev_end = np.concatenate([[-np.inf], re[:-1]])
    next_start = np.concatenate([rs[1:], [np.inf]])
    es = np.maximum(0, ps - rs) + np.maximum(0, np.minimum(prev_end, rs) - ps)
    ee = np.maximum(0, re - pe) + np.maximum(0, pe - np.maximum(next_start, re))
    return es, ee


def score(clips: list[dict]) -> dict[str, float]:
    """Each clip: pred_start, pred_end, found, ref_start, ref_end (seconds), dur."""
    asym, sym, worst, found = [], [], [], []
    for c in clips:
        rs, re = np.asarray(c["ref_start"]), np.asarray(c["ref_end"])
        ps, pe = fill_missing(
            np.asarray(c["pred_start"]), np.asarray(c["pred_end"]), np.asarray(c["found"]), c["dur"]
        )
        es, ee = asymmetric_errors(ps, pe, rs, re)
        asym += [es, ee]
        sym += [np.abs(ps - rs), np.abs(pe - re)]
        worst.append(np.maximum(es, ee))
        found.append(np.asarray(c["found"]))
    asym, sym, worst = np.concatenate(asym), np.concatenate(sym), np.concatenate(worst)
    r = np.concatenate(found).mean()
    out = {
        "mean_asym_ms": 1000 * asym.mean(),
        "mean_sym_ms": 1000 * sym.mean(),
        "f1": 2 * r / (1 + r),
        "words": len(worst),
    }
    for tol in TOLERANCES_MS:
        out[f"err>{tol}ms_%"] = 100 * (worst > tol / 1000).mean()
    return out
