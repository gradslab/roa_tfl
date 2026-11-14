r"""
Final test for the scenario bound (paper, Proposition 4): independent i.i.d. samples.  The function v and the levels c1, c2_old were
fixed before this test, so they are independent of its samples.  N i.i.d. uniform box samples are drawn; c2_iid is the
lowest v among the samples in {c1 <= v < c2_old} that fail the multi-horizon condition, and the reported level is
c2 = min(c2_old, c2_iid).  Samples with v >= c2_old cannot change c2, so they are drawn and counted but need no flow.
With P uniform on the box, P[ vol(failures in {c1 <= v < c2}) / vol(box) > eps ] <= (1 - eps)^N, and the same holds for
the reported c2 because lowering the level only shrinks the set.  eps = ln(1e6) / N gives confidence 1 - 1e-6.
"""
from __future__ import annotations

import math
import zlib

import torch

N_IID = 1000000


def seed_of(name):
    return 7000000 + zlib.crc32(name.encode()) % 1000000


@torch.no_grad()
def failures(v, step, znorm, x, W, horizons, z_tol):
    """Multi-horizon test of certify.two_stage: 1 = step leaves Omega_ad, 0 = no decrease at any horizon, -1 = passes."""
    y, ok = step(x, 1.0)
    dec = ok & (torch.nan_to_num(v(y), nan=1.0) >= W) & (znorm(x) > z_tol)
    for T in horizons:
        r = torch.nonzero(dec).squeeze(-1)
        if len(r) == 0:
            break
        yT, okT = step(x[r], T)
        dec[r] = ~(okT & (torch.nan_to_num(v(yT), nan=1.0) < W[r]))
    kind = torch.full_like(W, -1, dtype=torch.long)
    kind[dec] = 0
    kind[~ok] = 1
    return kind


@torch.no_grad()
def iid_test(v, inner, draw, step, znorm, dim, c1, c2_old, seed, n=N_IID, horizons=(2.0, 4.0, 8.0), chunk=200000,
             batch=20000, z_tol=1e-3):
    """v, inner: x -> values; draw: u in [0, 1)^dim -> box; step(x, T) -> (y, ok).  Returns the test record."""
    g = torch.Generator().manual_seed(int(seed))
    band_x, band_w, low_x, low_w = [], [], [], []
    done = 0
    while done < n:
        m = min(chunk, n - done)
        x = draw(torch.rand(m, dim, generator=g))
        W = torch.cat([torch.nan_to_num(v(c), nan=1.0) for c in x.split(batch)])
        b = (W >= c1) & (W < c2_old)
        band_x.append(x[b])
        band_w.append(W[b])
        I = torch.cat([inner(c) for c in x.split(batch)])
        low = (W < c1) & (I >= 1)
        low_x.append(x[low])
        low_w.append(W[low])
        done += m
    X, W = torch.cat(band_x), torch.cat(band_w)
    o = torch.argsort(W)
    X, W = X[o], W[o]
    c2_iid, n_tested, first = 1.0, 0, None
    for j in torch.arange(len(W), device=W.device).split(batch):
        kind = failures(v, step, znorm, X[j], W[j], horizons, z_tol)
        n_tested += len(j)
        bad = torch.nonzero(kind >= 0).squeeze(-1)
        if len(bad):
            i = bad[0]
            c2_iid = W[j][i].item()
            first = dict(v=c2_iid, kind=int(kind[i]), x=[float(t) for t in X[j][i].tolist()])
            break
    XL, WL = torch.cat(low_x), torch.cat(low_w)
    low_fail = 0
    for j in torch.arange(len(WL), device=WL.device).split(batch):
        low_fail += int((failures(v, step, znorm, XL[j], WL[j], horizons, z_tol) >= 0).sum())
    return dict(N=n, seed=int(seed), distribution="uniform on the box, i.i.d.", eps=math.log(1e6) / n, confidence=1 - 1e-6,
                c1=c1, c2_old=c2_old, c2_iid=c2_iid, c2=max(min(c2_old, c2_iid), c1), n_band=len(W), n_tested=n_tested,
                first_failure=first, n_below_c1_outside_inner=len(WL), n_below_c1_outside_inner_failed=low_fail)
